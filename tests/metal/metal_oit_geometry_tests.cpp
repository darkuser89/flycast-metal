// SPDX-License-Identifier: GPL-2.0-or-later
#include "rend/metal4/metal4_oit_geometry.h"
#include "rend/metal4/metal4_oit_plan.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <random>
using u32 = uint32_t;
using Matrix = std::array<std::array<float,4>,4>;
struct Vertex { float x,y,z; };
struct Vertices {
	std::vector<Vertex> data;
	mutable size_t reads = 0;
	size_t size() const { return data.size(); }
	const Vertex& operator[](size_t i) const { ++reads; return data[i]; }
};
struct Frame { Matrix ndc; u32 width, height; };
// Independent pre-optimization renderer implementation, retained as an oracle.
metal_oit::Tile reference(const Frame& frame, const Vertices& vertices, const u32 *index, bool naomi2 = false)
{
	const metal_oit::Tile whole{0,0,frame.width,frame.height};
				// Naomi2's shader transforms, or a noncanonical projection, need
				// the full screen bound. Positive Dreamcast W is constant one;
				// native-depth division cancels when forming screen X/Y as well.
				const auto& m = frame.ndc;
				bool bounded = !naomi2 && m[0][3] == 0 && m[1][3] == 0 && m[2][3] == 0 && m[3][3] == 1
					&& m[0][2] == 0 && m[1][2] == 0 && m[3][2] == 0 && std::isnormal(m[2][2]) && m[2][2] > 0;
				double left = frame.width, top = frame.height, right = 0, bottom = 0;
				for (int j = 0; j < 3 && bounded; ++j) {
					if (index[j] >= vertices.size()) { bounded = false; break; }
					const auto& v = vertices[index[j]];
					// A nonpositive/subnormal reciprocal can cross homogeneous W
					// during native-depth clipping; vertex X/Y no longer bounds it.
					if (!std::isnormal(v.z) || v.z <= 0) { bounded = false; break; }
					const double reciprocal = double(m[2][2]) * v.z;
					if (reciprocal < 1.e-12 || reciprocal > 1.e5) { bounded = false; break; }
					const double values[]{v.x, v.y, v.z, 1};
					double screen[2]{};
					for (int axis = 0; axis < 2; ++axis) {
						double value = 0, magnitude = 0;
						for (int k = 0; k < 4; ++k) { const double term = double(m[k][axis]) * values[k]; value += term; magnitude += std::abs(term); }
						// Clipped strips can have very large off-screen vertices.
						// Their rounding margin below remains conservative; only
						// float overflow, including native-depth division, requires
						// the full-screen fallback, not a fixed coordinate cutoff.
						const double safeMagnitude = double(std::numeric_limits<float>::max())
							* std::min(1.0, reciprocal) / 16;
						if (!std::isfinite(value) || !std::isfinite(magnitude) || magnitude > safeMagnitude) { bounded = false; break; }
						// Include raster subpixel rounding and float matrix error.
						const double size = axis == 0 ? frame.width : frame.height;
						const double margin = 4 + magnitude * std::numeric_limits<float>::epsilon() * 16 * size;
						screen[axis] = (value + 1) * size * .5;
						if (axis == 0) { left = std::min(left, screen[axis] - margin); right = std::max(right, screen[axis] + margin); }
						else { top = std::min(top, screen[axis] - margin); bottom = std::max(bottom, screen[axis] + margin); }
					}
				}
				if (!bounded) return whole;
				else {
					const u32 x = u32(std::clamp(std::floor(left), 0.0, double(frame.width)));
					const u32 y = u32(std::clamp(std::floor(top), 0.0, double(frame.height)));
					const u32 endX = u32(std::clamp(std::ceil(right), 0.0, double(frame.width)));
					const u32 endY = u32(std::clamp(std::ceil(bottom), 0.0, double(frame.height)));
					return {x,y,endX-x,endY-y};
				}
}
bool equal(metal_oit::Tile a,metal_oit::Tile b) { return a.x==b.x && a.y==b.y && a.width==b.width && a.height==b.height; }

Matrix identity()
{
	Matrix m{};
	for (unsigned i=0;i<4;++i) m[i][i]=1;
	return m;
}
using Position = std::array<float,4>;
// Deliberately independent scalar shader oracle. Exercise both unfused dots
// and fused dots: either contraction is permitted by the Metal shader build.
Position shaderMultiply(const Matrix& m,Position v,bool fused)
{
	Position out{};
	for (unsigned row=0;row<4;++row) {
		if (fused) {
			for (unsigned col=0;col<4;++col) out[row]=std::fma(m[col][row],v[col],out[row]);
		} else {
			volatile float products[4];
			for (unsigned col=0;col<4;++col) products[col]=m[col][row]*v[col];
			volatile float first=products[0]+products[1],second=first+products[2];
			out[row]=second+products[3];
		}
	}
	return out;
}
struct ScreenPoint { double x,y; };
ScreenPoint shaderNaomi2(const Frame& f,const Matrix& mv,const Matrix& projection,Vertex v,bool fused)
{
	auto p=shaderMultiply(projection,shaderMultiply(mv,{v.x,v.y,v.z,1},fused),fused);
	p={p[0]/p[3],p[1]/p[3],1.f/p[3],1};
	p=shaderMultiply(f.ndc,p,fused);
	// The vertex shader negates Y, then Metal's viewport reverses it again.
	return {(double(p[0])+1)*f.width*.5,(double(p[1])+1)*f.height*.5};
}
bool contains(metal_oit::Tile b,double x,double y)
{
	return x>=b.x && y>=b.y && x<=uint64_t(b.x)+b.width && y<=uint64_t(b.y)+b.height;
}
void verifyNaomi2Coverage(const Frame& f,const Matrix& mv,const Matrix& projection,
		const Vertices& vertices,const u32 *indices,metal_oit::Tile bounds,uint64_t& samples)
{
	for (bool fused:{false,true}) {
		ScreenPoint p[3];
		for (unsigned i=0;i<3;++i) p[i]=shaderNaomi2(f,mv,projection,vertices[indices[i]],fused);
		if (!std::all_of(p,p+3,[](auto v){return std::isfinite(v.x)&&std::isfinite(v.y);})) continue;
		// All rasterized positions are convex combinations of the shader's
		// already divided positions (the final clip W is one). This samples
		// edges/interiors, including triangles whose vertices all lie offscreen.
		for (unsigned a=0;a<=12;++a) for (unsigned b=0;b<=12-a;++b) {
			const double x=(p[0].x*a+p[1].x*b+p[2].x*(12-a-b))/12;
			const double y=(p[0].y*a+p[1].y*b+p[2].y*(12-a-b))/12;
			if (x>=0 && y>=0 && x<f.width && y<f.height) {assert(contains(bounds,x,y));++samples;}
		}
		const auto edge=[](ScreenPoint a,ScreenPoint b,double x,double y) {
			return (b.x-a.x)*(y-a.y)-(b.y-a.y)*(x-a.x);
		};
		const double area=edge(p[0],p[1],p[2].x,p[2].y);
		if (std::abs(area)<1.e-12) continue;
		// Independent point-in-triangle raster samples at pixel centers.
		for (u32 y=0;y<f.height;y+=std::max(1u,f.height/17))
			for (u32 x=0;x<f.width;x+=std::max(1u,f.width/19)) {
				const double a=edge(p[0],p[1],x+.5,y+.5),b=edge(p[1],p[2],x+.5,y+.5),c=edge(p[2],p[0],x+.5,y+.5);
				if ((a>=0 && b>=0 && c>=0)||(a<=0 && b<=0 && c<=0)) {
					assert(contains(bounds,x+.5,y+.5));++samples;
				}
			}
	}
}
void validateNaomi2Bounds()
{
	std::mt19937 random(0x4e324f49);
	Frame f{};f.width=2560;f.height=1920;
	f.ndc[0][0]=2.f/640;f.ndc[1][1]=-2.f/480;f.ndc[2][2]=1;
	f.ndc[3][0]=-1;f.ndc[3][1]=1;f.ndc[3][3]=1;
	Matrix projection{};
	projection[0][0]=320;projection[1][1]=240;
	projection[2][0]=320;projection[2][1]=240;projection[2][3]=1;
	Vertices vertices;vertices.data.resize(3);u32 index[]{0,1,2};
	uint64_t samples=0;size_t nonFull=0;
	for (unsigned fixture=0;fixture<3000;++fixture) {
		f.width=1+random()%7680;f.height=1+random()%4320;
		Matrix mv=identity(),proj=projection;
		const float angle=float(random()%10000)*.00062831853f,c=std::cos(angle),s=std::sin(angle);
		mv[0][0]=c;mv[0][1]=s;mv[1][0]=-s;mv[1][1]=c;
		mv[3][0]=float(int(random()%600)-300)*.01f;
		mv[3][1]=float(int(random()%600)-300)*.01f;
		mv[3][2]=1+float(random()%800)*.01f;
		proj[1][0]=float(int(random()%200)-100)*.1f;
		proj[0][1]=float(int(random()%200)-100)*.1f;
		proj[0][3]=float(int(random()%200)-100)*.001f;
		proj[1][3]=float(int(random()%200)-100)*.001f;
		for (auto& v:vertices.data) v={float(int(random()%1600)-800)*.01f,float(int(random()%1200)-600)*.01f,float(random()%600)*.01f};
		// Near-singular denominators, cancellation, negative W, offscreen
		// coordinates and very large products must remain conservative.
		if(fixture%7==0)for(auto& v:vertices.data)v.z=-mv[3][2]+float(int(random()%3)-1)*1.e-6f;
		if(fixture%11==0)proj[2][3]=-1;
		if(fixture%13==0){mv[0][0]=1.e12f;mv[3][0]=-1.e12f;vertices.data[0].x=1;}
		metal_oit::GeometryBounds geometry(f.ndc,f.width,f.height,vertices.size());
		geometry.setNaomi2Projection(mv,proj);
		const auto bounds=geometry.triangleNaomi2(index,vertices);
		nonFull+=!equal(bounds,{0,0,f.width,f.height});
		verifyNaomi2Coverage(f,mv,proj,vertices,index,bounds,samples);
	}
	assert(nonFull>1000 && samples>100000);
	f.width=2560;f.height=1920;
	vertices.data={{-.03f,-.03f,1},{.03f,-.03f,1},{0,.03f,1}};
	metal_oit::GeometryBounds geometry(f.ndc,f.width,f.height,vertices.size());
	const auto mv=identity();geometry.setNaomi2Projection(mv,projection);
	const auto original=geometry.triangleNaomi2(index,vertices);
	assert(original.width<f.width/4 && original.height<f.height/4);
	Matrix translated=mv;translated[3][0]=1;
	geometry.setNaomi2Projection(translated,projection);
	const auto moved=geometry.triangleNaomi2(index,vertices);
	assert(moved.x>original.x+f.width/4);
	// Switching back to PVR must not reuse the N2 projection/cache entries.
	assert(equal(geometry.triangle(index,vertices),reference(f,vertices,index)));
	geometry.setNaomi2Projection(mv,projection);
	assert(equal(geometry.triangleNaomi2(index,vertices),original));
	vertices.reads=0;
	for(unsigned i=0;i<100000;++i)geometry.triangleNaomi2(index,vertices);
	assert(vertices.reads==0); // All three points are already in the strip cache.
	vertices.data[0].x=std::numeric_limits<float>::quiet_NaN();
	geometry.setNaomi2Projection(mv,projection);
	assert(equal(geometry.triangleNaomi2(index,vertices),{0,0,f.width,f.height}));
	vertices.data[0]={-.03f,-.03f,0};
	geometry.setNaomi2Projection(mv,projection);
	assert(equal(geometry.triangleNaomi2(index,vertices),{0,0,f.width,f.height}));
	vertices.data[0]={-.03f,-.03f,1};
	Matrix invalid=projection;invalid[2][3]=0;
	geometry.setNaomi2Projection(mv,invalid);
	assert(equal(geometry.triangleNaomi2(index,vertices),{0,0,f.width,f.height}));
	invalid=projection;invalid[0][0]=std::numeric_limits<float>::infinity();
	geometry.setNaomi2Projection(mv,invalid);
	assert(equal(geometry.triangleNaomi2(index,vertices),{0,0,f.width,f.height}));
	geometry.setNaomi2Projection(mv,projection);u32 bad[]{0,1,3};
	assert(equal(geometry.triangleNaomi2(bad,vertices),{0,0,f.width,f.height}));
	metal_oit::GeometryBounds disabled(f.ndc,f.width,f.height,vertices.size(),false);
	disabled.setNaomi2Projection(mv,projection);
	assert(equal(disabled.triangleNaomi2(index,vertices),{0,0,f.width,f.height}));
	// All indices collide in the small direct-mapped strip cache. The first
	// projected corner must not be overwritten while calculating the third.
	vertices.data.resize(9);
	vertices.data[0]={-.6f,-.4f,1};vertices.data[4]={.6f,-.4f,1};vertices.data[8]={0,.4f,1};
	u32 collision[]{0,4,8};geometry.setNaomi2Projection(mv,projection);
	verifyNaomi2Coverage(f,mv,projection,vertices,collision,geometry.triangleNaomi2(collision,vertices),samples);
	// Frame reset invalidates both the projection selection and dimensions.
	geometry.reset(f.ndc,640,480,vertices.size());
	assert(equal(geometry.triangleNaomi2(collision,vertices),{0,0,640,480}));
	geometry.setNaomi2Projection(mv,projection);
	verifyNaomi2Coverage({f.ndc,640,480},mv,projection,vertices,collision,geometry.triangleNaomi2(collision,vertices),samples);
	// 20,000 small projected triangles spread across a 2560x1920 scene. The
	// old whole-frame fallback requires 2048 tiles even with a 1-GiB node
	// buffer; the same coverage proof must now fit without replaying them.
	vertices.data.resize(60000);std::vector<metal_oit::Tile> before,after;
	uint64_t beforeArea=0,afterArea=0;
	geometry.reset(f.ndc,f.width,f.height,vertices.size());
	geometry.setNaomi2Projection(mv,projection);
	for(u32 triangle=0;triangle<20000;++triangle) {
		const float z=1+float(triangle%13),cx=(float(triangle%200)+.5f)/200*640,cy=(float(triangle/200)+.5f)/100*480;
		const float x=(cx-320)*z/320,y=(cy-240)*z/240;
		const size_t base=triangle*3;
		vertices.data[base]={x-.001f*z,y-.001f*z,z};vertices.data[base+1]={x+.001f*z,y-.001f*z,z};vertices.data[base+2]={x,y+.001f*z,z};
		u32 indices[]{u32(base),u32(base+1),u32(base+2)};
		const auto bounds=geometry.triangleNaomi2(indices,vertices);
		before.push_back({0,0,f.width,f.height});after.push_back(bounds);
		metal_oit::TilePlanner::addCoverage(beforeArea,before.back());metal_oit::TilePlanner::addCoverage(afterArea,bounds);
		for(bool fused:{false,true})for(auto i:indices) {
			const auto p=shaderNaomi2(f,mv,projection,vertices[i],fused);assert(contains(bounds,p.x,p.y));
		}
	}
	metal_oit::TilePlanner planner;const uint64_t capacity=53687090;
	const size_t oldTiles=planner.plan(f.width,f.height,before,capacity,beforeArea).size();
	const size_t newTiles=planner.plan(f.width,f.height,after,capacity,afterArea).size();
	assert(oldTiles==2048 && newTiles==1);
	printf("PASS: NAOMI 2: 3000 independent float/FMA shader-projection fixtures, %llu coverage samples, %zu non-full bounds; matrix/PVR/invalid fallbacks; 20000-triangle synthetic scene tiles %zu -> %zu\n",(unsigned long long)samples,nonFull,oldTiles,newTiles);
	// Diagnostic cost only, with preallocated inputs/results and one matrix
	// selection per triangle (the worst case of one triangle per polygon).
	// Do not time fixture creation, the independent oracle or old full bounds.
	std::array<double,7> planningMs;
	for(auto& elapsed:planningMs) {
		uint64_t area=0;
		const auto started=std::chrono::steady_clock::now();
		for(u32 triangle=0;triangle<20000;++triangle) {
			const u32 first=triangle*3,indices[]{first,first+1,first+2};
			geometry.setNaomi2Projection(mv,projection);
			after[triangle]=geometry.triangleNaomi2(indices,vertices);
			metal_oit::TilePlanner::addCoverage(area,after[triangle]);
		}
		assert(planner.plan(f.width,f.height,after,capacity,area).size()==1);
		elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
	}
	std::sort(planningMs.begin(),planningMs.end());
	printf("INFO: 20000-triangle projected-bounds + tile planning median %.3f ms (7 runs, one matrix selection per triangle; synthetic CPU cost only)\n",planningMs[3]);
}
int main()
{
	const auto testStarted=std::chrono::steady_clock::now();
	std::mt19937 random(48169);
	Frame f{}; f.width=1920;f.height=1080;
	f.ndc[0][0]=2.f/640;f.ndc[1][1]=-2.f/480;f.ndc[2][2]=1;
	f.ndc[3][0]=-1;f.ndc[3][1]=1;f.ndc[3][3]=1;
	Vertices v;v.data.resize(30);
	for (unsigned fixture=0;fixture<5000;++fixture) {
		for (auto& p:v.data) {
			p.x=float(int(random()%32000)-16000);p.y=float(int(random()%32000)-16000);
			p.z=std::ldexp(float(1+random()%1000)/1000,int(random()%70)-35);
		}
		if(fixture%7==0)v.data[fixture%v.size()].x=std::numeric_limits<float>::infinity();
		if(fixture%11==0)v.data[fixture%v.size()].z=-1;
		if(fixture%13==0)v.data[fixture%v.size()].y=std::numeric_limits<float>::quiet_NaN();
		if(fixture%17==0)v.data[fixture%v.size()].z=std::numeric_limits<float>::denorm_min();
		if(fixture%31==0)v.data[fixture%v.size()].x=1.e30f;
		if(fixture%37==0)v.data[fixture%v.size()].y=std::numeric_limits<float>::max();
		Matrix matrix=f.ndc;
		if(fixture&1) { matrix[0][1]=.004f;matrix[1][0]=-.001f;matrix[2][0]=.03f; }
		if(fixture%3==0)matrix[0][0]=-matrix[0][0];
		if(fixture%19==0)matrix[0][3]=.1f;
		if(fixture%23==0)matrix[2][2]=std::numeric_limits<float>::quiet_NaN();
		Frame frame{matrix,1+random()%7680,1+random()%4320};
		metal_oit::GeometryBounds geometry(matrix,frame.width,frame.height,v.size());
		metal_oit::Tile polygon{};
		std::vector<metal_oit::Tile> triangles;
		for(unsigned i=0;i<32;++i) {
			u32 indices[]{random()%32,random()%32,random()%32};
			const bool naomi2=i%13==0;
			const auto expected=reference(frame,v,indices,naomi2), actual=geometry.triangle(indices,v,naomi2);
			assert(equal(expected,actual));
			if(actual.width && actual.height) {triangles.push_back(actual);polygon=metal_oit::coverageUnion(polygon,actual);}
		}
		// Every contributing primitive must survive polygon selection for every
		// disjoint tile, including zero-area/off-screen primitives in the union.
		for(u32 y=0;y<frame.height;y+=128)for(u32 x=0;x<frame.width;x+=128) {
			const metal_oit::Tile tile{x,y,std::min(128u,frame.width-x),std::min(128u,frame.height-y)};
			const auto poly=metal_oit::intersection(tile,polygon);
			if(!poly.width || !poly.height)for(auto triangle:triangles) {
				const auto overlap=metal_oit::intersection(tile,triangle);assert(!overlap.width || !overlap.height);
			}
		}
	}
	v.data={{1,1,1},{2,2,1},{3,1,1}};v.reads=0;
	metal_oit::GeometryBounds geometry(f.ndc,f.width,f.height,v.size());u32 index[]{0,1,2};
	for(unsigned i=0;i<100000;++i)geometry.triangle(index,v);
	assert(v.reads==3);
	// Reuse the same allocation across logical passes; explicitly reset it
	// when the CPU vertex data or projection changes for a later render.
	v.data[0].x=100; v.data[0].z=1; v.reads=0;
	geometry.reset(f.ndc,f.width,f.height,v.size());
	assert(equal(geometry.triangle(index,v),reference(f,v,index)));
	v.reads=0;
	for(unsigned pass=0;pass<4;++pass)geometry.triangle(index,v);
	assert(v.reads==0);
	v.data[0].z=0;
	geometry.reset(f.ndc,f.width,f.height,v.size());
	assert(equal(geometry.triangle(index,v),{0,0,f.width,f.height}));
	v.data[0].z=1;
	geometry.reset(f.ndc,13,9,v.size(),false);
	assert(equal(geometry.triangle(index,v),{0,0,13,9}));
	// A new render/projection must rebuild the proof, including failed bounds.
	v.data[0].z=0;v.reads=0;
	metal_oit::GeometryBounds invalid(f.ndc,f.width,f.height,v.size());
	for(unsigned i=0;i<100000;++i)assert(equal(invalid.triangle(index,v),{0,0,f.width,f.height}));
	assert(v.reads==3);
	metal_oit::GeometryBounds disabled(f.ndc,f.width,f.height,v.size(),false);
	assert(equal(disabled.triangle(index,v),{0,0,f.width,f.height}));
	puts("PASS: 160000 exact comparisons against old geometry bounds, 5000 polygon/tile coverage fixtures; 100000 shared triangles read only 3 vertices; invalid/projection fallbacks preserved");
	validateNaomi2Bounds();
	printf("INFO: geometry regression total %.3f ms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-testStarted).count());
}
