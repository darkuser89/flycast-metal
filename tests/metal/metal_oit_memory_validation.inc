// SPDX-License-Identifier: GPL-2.0-or-later
// Explicit CLI-only long-running check of production OIT capture/modifiers/resolve.
void MetalRenderer::validateSyntheticOITMemory()
{
	const auto require=[](bool value,const char *message){if(!value)throw RendererException(message);};
	const auto footprint=[&] {
		task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
		require(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS,
			"Cannot read OIT process footprint");return uint64_t(info.phys_footprint);
	};
	const auto oldSorting=config::PerStripSorting.get();const auto oldLayers=config::PerPixelLayers.get();
	config::PerStripSorting=true;config::PerPixelLayers=64;
	struct Restore{bool sorting;int layers;~Restore(){config::PerStripSorting=sorting;config::PerPixelLayers=layers;}} restore{oldSorting,oldLayers};
	disableDenseOIT=disablePackedOIT=true;measureOITStorage=true;
	rend_context fixture{};rc=&fixture;fixture.clearFramebuffer=true;
	fixture.globClip={640,480};fixture.tileClip=fixture.fbClip=Rect({0,0},{640,480});
	fixture.framebufferWidth=160;fixture.framebufferHeight=120;
	fixture.scaler_ctl.vscalefactor=1024;fixture.fog_clamp_max.full=~0u;
	constexpr unsigned quads=49;
	for(unsigned q=0;q<quads;++q) {
		for(unsigned i=0;i<4;++i) {
			Vertex v{};v.x=i&1?640:0;v.y=i&2?480:0;v.z=q?1.5f+q*.01f:1.f;
			v.col[0]=q?101:37;v.col[1]=q?73:149;v.col[2]=q?47:199;v.col[3]=q?31:255;
			fixture.verts.push_back(v);fixture.idx.push_back(q*4+i);
		}
		PolyParam poly;poly.init();poly.first=q*4;poly.count=4;poly.pcw.Gouraud=1;
		poly.isp.DepthMode=7;poly.tsp.FogCtrl=2;poly.tsp.SrcInstr=q?4:1;
		if(q){poly.isp.ZWriteDis=1;poly.pcw.Shadow=1;poly.tsp.UseAlpha=1;poly.tsp.DstInstr=5;fixture.global_param_tr.push_back(poly);}
		else fixture.global_param_op.push_back(poly);
	}
	fixture.modtrig={{0,0,4,320,0,4,0,480,4},{320,0,4,320,480,4,0,480,4}};
	ModifierVolumeParam volume;volume.init();volume.count=2;volume.isp.DepthMode=1;volume.isp.VolumeLast=1;
	fixture.global_param_mvo_tr={volume};RenderPass pass{};pass.op_count=1;pass.tr_count=quads-1;pass.mvo_tr_count=1;pass.autosort=true;
	fixture.render_passes={pass};u8 zero[256]{};
	fog.tex_type=TextureType::_8;fog.UploadToGPU(128,2,zero,false);
	palette.tex_type=TextureType::_8888;palette.UploadToGPU(1,1,zero,false);
	context()->flushTextureUpdates();context()->waitIdle();
	// The app's OSD calls the real Metal ImGui driver during Present. A no-op
	// GUI stub would miss work encoded after beginPresentation's inner pool.
	u32 white=~0u;auto uiTexture=imguiDriver->updateTexture("__metal_oit_memory_osd",reinterpret_cast<u8 *>(&white),1,1,true);
	ImDrawList list(ImGui::GetDrawListSharedData());list.VtxBuffer.resize(4);list.IdxBuffer.resize(6);
	for(unsigned i=0;i<4;++i)list.VtxBuffer[i]={ImVec2(i&1?56.f:8.f,i&2?40.f:8.f),ImVec2(.5f,.5f),IM_COL32(255,0,0,128)};
	const ImDrawIdx ids[]={0,1,2,2,1,3};std::copy_n(ids,6,list.IdxBuffer.Data);
	list._VtxWritePtr=list.VtxBuffer.Data+4;list._IdxWritePtr=list.IdxBuffer.Data+6;list._VtxCurrentIdx=4;
	ImDrawCmd draw;draw.ElemCount=6;draw.ClipRect=ImVec4(8,8,56,40);draw.TexRef=ImTextureRef(uiTexture);list.CmdBuffer.push_back(draw);
	ImDrawData ui;ui.Valid=true;ui.DisplaySize=ImVec2(64,48);ui.FramebufferScale=ImVec2(1,1);ui.AddDrawList(&list);
	struct RestoreOSD{std::function<void()> callback;~RestoreOSD(){metalOffscreenDrawOSD=std::move(callback);}} restoreOSD{std::move(metalOffscreenDrawOSD)};
	metalOffscreenDrawOSD=[&]{imguiDriver->renderDrawData(&ui,false);};
	const auto hash=[](const std::vector<u8>& pixels){uint64_t value=1469598103934665603ULL;for(auto byte:pixels){value^=byte;value*=1099511628211ULL;}return value;};
	uint64_t baseline=0,peak=0;std::vector<u8> reference,presentedReference;
	std::array<FCMetal4Commands *,MetalContext::RenderBatchSlots> pending{};
	for(unsigned frameIndex=0;frameIndex<256;++frameIndex) {
		FCMetal4Commands *command;
		// Match the app's pool boundary: Render drains its own pool before
		// Present commits the deferred batch. A whole-frame test pool would
		// hide autoreleased native encoders retained until the outer run loop.
		@autoreleasepool {
			measuredOITTiles=0;oitPolygonDraws=0;
			metalOffscreenArmCommand(MetalOffscreenFault::None,nil);
			require(Render(),"OIT memory render failed");command=metalOffscreenLastCommand();
			require(command!=nil,"OIT memory command was not observed");
		}
		if(metalOffscreenOITMemoryUnpooledControl) {
			// Explicit negative control: the exact same complete frame body,
			// bypassing only production Present's new autorelease-pool scope.
			clearLastFrame=false;
			require(presentFrame(),"OIT memory unpooled presentation failed");
		} else {
			clearLastFrame=false;
			require(Present(),"OIT memory production presentation failed");
		}
		imguiDriver->present();
		if(!metalOffscreenOITMemoryAsync)[command waitUntilCompleted];
		auto& previous=pending[frameIndex%pending.size()];
		if(previous)require(previous.status==FCMetal4StatusCompleted && !previous.error,"OIT memory prior GPU submission failed");
		previous=command;
		require(command.status>=FCMetal4StatusCommitted && !command.error,"OIT memory GPU submission failed");
		if(frameIndex%32==31) {
			@autoreleasepool {
				std::vector<u8> pixels;context()->readTexture(screen,pixels);
				if(reference.empty())reference=pixels;else require(reference==pixels,"OIT memory stress changed an output pixel");
				auto drawable=metalOffscreenTakeDrawableTexture();require(drawable!=nil,"OIT memory presentation had no offscreen output");
				context()->readTexture(drawable,pixels);
				if(presentedReference.empty())presentedReference=pixels;else require(presentedReference==pixels,"OIT memory stress changed a presentation pixel");
				if(frameIndex==31)NOTICE_LOG(RENDERER,"Metal OIT memory reference OIT_hash=%016llx presentation_hash=%016llx",
					(unsigned long long)hash(reference),(unsigned long long)hash(presentedReference));
			}
		}
		if(frameIndex==31)baseline=footprint();
		if(frameIndex>=31)peak=std::max(peak,footprint());
		if(peak>baseline+128_MB) {
			NOTICE_LOG(RENDERER,"Metal OIT memory growth stopped frame=%u warm=%llu peak=%llu unpooled_control=%d",frameIndex+1,
				(unsigned long long)baseline,(unsigned long long)peak,int(metalOffscreenOITMemoryUnpooledControl));
			throw RendererException("OIT live GPU resources grew more than 128 MiB after warmup");
		}
		if(frameIndex%32==31) {
			NOTICE_LOG(RENDERER,"Metal OIT memory frame=%u footprint=%llu metal=%llu tiles=%llu draws=%llu unpooled_control=%d",frameIndex+1,
				(unsigned long long)footprint(),(unsigned long long)context()->device().currentAllocatedSize,
				(unsigned long long)measuredOITTiles,(unsigned long long)oitPolygonDraws,int(metalOffscreenOITMemoryUnpooledControl));
		}
	}
	context()->waitIdle();
	for(auto command:pending)require(command.status==FCMetal4StatusCompleted && !command.error,"OIT memory final GPU submission failed");
	rc=nullptr;NOTICE_LOG(RENDERER,"Metal validation PASS: 256 production OIT memory frames with Present outside the Render pool, 8 exact OIT and 8 exact presentation readbacks; warm footprint=%llu peak=%llu OIT_hash=%016llx presentation_hash=%016llx",
		(unsigned long long)baseline,(unsigned long long)peak,(unsigned long long)hash(reference),(unsigned long long)hash(presentedReference));
}
