// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "metal4_oit_tiling.h"
#include <array>
#include <cmath>
#include <limits>

namespace metal_oit {
// Union of nonempty primitive coverage; never use this to join individual
// off-screen vertices, whose clamped empty extents can still span a triangle.
inline Tile coverageUnion(Tile a, Tile b)
{
	if (!a.width || !a.height) return b;
	if (!b.width || !b.height) return a;
	const uint32_t x = std::min(a.x, b.x), y = std::min(a.y, b.y);
	return {x, y, uint32_t(std::max(uint64_t(a.x) + a.width, uint64_t(b.x) + b.width) - x),
		uint32_t(std::max(uint64_t(a.y) + a.height, uint64_t(b.y) + b.height) - y)};
}

// Lifetime: one immutable vertex array/projection. Reset before a new render.
// Reused strip vertices (also across passes) are transformed once, including
// failed bounds. Noncanonical projections stay unbounded. Naomi2 has a
// separate, polygon-local projection cache because its matrices can change
// even when a later polygon references the same vertex indices.
class GeometryBounds {
	struct Point {
		double left = 0, top = 0, right = 0, bottom = 0;
		enum { Unknown, Bounded, Unbounded } state = Unknown;
	};
	std::vector<Point> points;
	std::array<std::array<float, 4>, 4> matrix;
	Tile whole{};
	bool canonical = false;
	struct Naomi2Point { uint64_t generation = 0; uint32_t index = ~0u; Point point; };
	std::array<Naomi2Point, 64> naomi2Points{};
	uint64_t naomi2Generation = 0;
	bool naomi2Configured = false;

	struct Interval { double low, high; };
	struct IntervalRow {
		std::array<Interval,4> coefficients;
		std::array<unsigned,4> columns;
		unsigned count=0;
	};
	struct IntervalMatrix {
		// Store rows in consumption order. Coefficients and their optional
		// subnormal-to-zero range do not change while vertices are projected.
		std::array<IntervalRow,4> rows;
		bool valid = false;
	};
	IntervalMatrix naomi2MV, naomi2Projection, naomi2NDC;
	static Interval input(float value)
	{
		// Metal may flush subnormal operands to zero. Include either result,
		// including for a subnormal matrix coefficient times a large vertex.
		if (std::abs(value) < std::numeric_limits<float>::min())
			return {std::min(0.0, double(value)), std::max(0.0, double(value))};
		return {value, value};
	}
	static bool finite(Interval value)
	{
		return std::isfinite(value.low) && std::isfinite(value.high)
			&& value.low <= value.high;
	}
	static Interval product(Interval a, Interval b)
	{
		// Normal float coefficients have identical endpoints. The old four
		// products were two duplicates; preserve their exact bounds with two.
		if(a.low==a.high) {
			if(a.low<0)return {a.low*b.high,a.low*b.low};
			return {a.low*b.low,a.low*b.high};
		}
		const double values[]{a.low*b.low, a.low*b.high, a.high*b.low, a.high*b.high};
		return {*std::min_element(values, values+4), *std::max_element(values, values+4)};
	}
	template<class Matrix> static void prepareMatrix(IntervalMatrix& destination,const Matrix& source)
	{
		destination.valid=true;
		for(unsigned axis=0;axis<4;++axis) {
			auto& row=destination.rows[axis];row.count=0;
			for(unsigned k=0;k<4;++k) {
				if(!std::isfinite(source[k][axis]))destination.valid=false;
				const auto coefficient=input(source[k][axis]);
				// Source intervals have already been checked finite. Zero
				// contributes neither interval extent nor rounding magnitude;
				// preserve the order of all remaining terms and each row's
				// error floor, including completely zero rows.
				if(coefficient.low==0 && coefficient.high==0)continue;
				row.coefficients[row.count]=coefficient;row.columns[row.count++]=k;
			}
		}
	}
	static bool transform(const IntervalMatrix& m,
		const std::array<Interval, 4>& source, std::array<Interval, 4>& destination)
	{
		if(!m.valid)return false;
		for (unsigned axis=0;axis<4;++axis) {
			Interval sum{0,0};
			double magnitude=0;
			const auto& row=m.rows[axis];
			for(unsigned k=0;k<row.count;++k) {
				const auto term=product(row.coefficients[k],source[row.columns[k]]);
				if(!finite(term))return false;
				sum.low+=term.low;sum.high+=term.high;
				magnitude+=std::max(std::abs(term.low),std::abs(term.high));
			}
			// Bound either separate float multiply/adds or contracted FMAs.
			// Propagate the preceding stage's interval instead of precombining
			// matrices: cancellation before the shader's division matters.
			// Reject possible intermediate overflow even if the sum cancels.
			if(!std::isfinite(magnitude) || magnitude>double(std::numeric_limits<float>::max())/4)return false;
			const double error=magnitude*std::numeric_limits<float>::epsilon()*16
				+double(std::numeric_limits<float>::min())*16;
			destination[axis]={sum.low-error,sum.high+error};
			if(!finite(destination[axis]))return false;
		}
		return true;
	}
	static bool divide(Interval numerator,Interval denominator,Interval& result)
	{
		if(!finite(denominator) || (denominator.low<=0 && denominator.high>=0))return false;
		const double values[]{numerator.low/denominator.low,numerator.low/denominator.high,
			numerator.high/denominator.low,numerator.high/denominator.high};
		result={*std::min_element(values,values+4),*std::max_element(values,values+4)};
		const double magnitude=std::max(std::abs(result.low),std::abs(result.high));
		if(!finite(result) || magnitude>double(std::numeric_limits<float>::max())/4)return false;
		const double error=magnitude*std::numeric_limits<float>::epsilon()*16
			+double(std::numeric_limits<float>::min())*16;
		result.low-=error;result.high+=error;
		return true;
	}
	template<class Vertices> const Point *projectNaomi2(uint32_t index,const Vertices& vertices)
	{
		if(index>=vertices.size())return nullptr;
		auto& cached=naomi2Points[index%naomi2Points.size()];
		if(cached.generation!=naomi2Generation || cached.index!=index) {
			cached.generation=naomi2Generation;cached.index=index;cached.point={};
		}
		auto& p=cached.point;
		if(p.state!=Point::Unknown)return p.state==Point::Bounded ? &p : nullptr;
		p.state=Point::Unbounded;
		const auto& v=vertices[index];
		std::array<Interval,4> source{input(v.x),input(v.y),input(v.z),Interval{1,1}},view,clip,projected,ndc;
		for(const auto& value:source)if(!finite(value))return nullptr;
		if(!transform(naomi2MV,source,view) || !transform(naomi2Projection,view,clip))return nullptr;
		// Match pvrVertex: projection.xy/W, reciprocal W, then a new W=1.
		// In particular, a finite negative projection W is not ordinary
		// homogeneous clipping: the shader already divided before rasterizing.
		if(!divide(clip[0],clip[3],projected[0]) || !divide(clip[1],clip[3],projected[1])
			|| !divide({1,1},clip[3],projected[2]))return nullptr;
		projected[3]={1,1};
		if(!transform(naomi2NDC,projected,ndc))return nullptr;
		// Shader Y negation and Metal's viewport Y inversion cancel, as in
		// the ordinary PVR bounds. Four pixels cover raster subpixel rounding.
		p.left=(ndc[0].low+1)*whole.width*.5-4;
		p.right=(ndc[0].high+1)*whole.width*.5+4;
		p.top=(ndc[1].low+1)*whole.height*.5-4;
		p.bottom=(ndc[1].high+1)*whole.height*.5+4;
		p.state=Point::Bounded;
		return &p;
	}
	Tile bounds(const Point& a,const Point& b,const Point& c) const
	{
		const uint32_t x=uint32_t(std::clamp(std::floor(std::min({a.left,b.left,c.left})),0.0,double(whole.width)));
		const uint32_t y=uint32_t(std::clamp(std::floor(std::min({a.top,b.top,c.top})),0.0,double(whole.height)));
		const uint32_t endX=uint32_t(std::clamp(std::ceil(std::max({a.right,b.right,c.right})),0.0,double(whole.width)));
		const uint32_t endY=uint32_t(std::clamp(std::ceil(std::max({a.bottom,b.bottom,c.bottom})),0.0,double(whole.height)));
		return {x,y,endX-x,endY-y};
	}

	template<class Vertices> const Point *project(uint32_t index, const Vertices& vertices)
	{
		if (index >= vertices.size()) return nullptr;
		auto& p = points[index];
		if (p.state != Point::Unknown) return p.state == Point::Bounded ? &p : nullptr;
		p.state = Point::Unbounded;
		const auto& v = vertices[index];
		if (!std::isnormal(v.z) || v.z <= 0) return nullptr;
		const double reciprocal = double(matrix[2][2]) * v.z;
		if (reciprocal < 1.e-12 || reciprocal > 1.e5) return nullptr;
		const double values[]{v.x, v.y, v.z, 1};
		for (int axis = 0; axis < 2; ++axis) {
			double value = 0, magnitude = 0;
			for (int k = 0; k < 4; ++k) {
				const double term = double(matrix[k][axis]) * values[k];
				value += term; magnitude += std::abs(term);
			}
			// Preserve the original bounds' float overflow and rounding margins,
			// including native-depth division and large off-screen coordinates.
			const double safeMagnitude = double(std::numeric_limits<float>::max()) * std::min(1.0, reciprocal) / 16;
			if (!std::isfinite(value) || !std::isfinite(magnitude) || magnitude > safeMagnitude) return nullptr;
			const double size = axis == 0 ? whole.width : whole.height;
			const double margin = 4 + magnitude * std::numeric_limits<float>::epsilon() * 16 * size;
			const double screen = (value + 1) * size * .5;
			if (axis == 0) { p.left = screen - margin; p.right = screen + margin; }
			else { p.top = screen - margin; p.bottom = screen + margin; }
		}
		p.state = Point::Bounded;
		return &p;
	}
public:
	GeometryBounds() = default;
	template<class Matrix> GeometryBounds(const Matrix& m, uint32_t width, uint32_t height, size_t vertexCount, bool enabled = true)
	{ reset(m,width,height,vertexCount,enabled); }
	template<class Matrix> void reset(const Matrix& m, uint32_t width, uint32_t height, size_t vertexCount, bool enabled = true)
	{
		whole = {0,0,width,height};
		canonical = enabled && m[0][3] == 0 && m[1][3] == 0 && m[2][3] == 0 && m[3][3] == 1
			&& m[0][2] == 0 && m[1][2] == 0 && m[3][2] == 0 && std::isnormal(m[2][2]) && m[2][2] > 0;
		for (int k = 0; k < 4; ++k) for (int axis = 0; axis < 4; ++axis) matrix[k][axis] = m[k][axis];
		prepareMatrix(naomi2NDC,matrix);
		points.assign(canonical ? vertexCount : 0,{});
		naomi2Configured=false;
	}
	// Call for each polygon, using the exact model-view/projection values
	// uploaded to its Naomi2 uniforms. A generation invalidates the bounded
	// strip cache without clearing it for every short polygon.
	template<class Matrix> void setNaomi2Projection(const Matrix& mv,const Matrix& projection)
	{
		prepareMatrix(naomi2MV,mv);prepareMatrix(naomi2Projection,projection);
		if(++naomi2Generation==0) {
			for(auto& p:naomi2Points)p={};
			naomi2Generation=1;
		}
		naomi2Configured=true;
	}
	template<class Vertices> Tile triangleNaomi2(const uint32_t *index,const Vertices& vertices)
	{
		if(!canonical || !naomi2Configured)return whole;
		// Copy each result: direct-mapped cache collisions must not change an
		// earlier point while the next vertex is projected.
		Point p[3];
		for(unsigned i=0;i<3;++i) {
			const Point *point=projectNaomi2(index[i],vertices);
			if(!point)return whole;
			p[i]=*point;
		}
		return bounds(p[0],p[1],p[2]);
	}
	template<class Vertices> Tile triangle(const uint32_t *index, const Vertices& vertices, bool naomi2 = false)
	{
		if (!canonical || naomi2) return whole;
		const Point *a = project(index[0], vertices), *b = project(index[1], vertices), *c = project(index[2], vertices);
		if (!a || !b || !c) return whole;
		return bounds(*a,*b,*c);
	}
};
}
