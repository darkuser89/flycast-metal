// SPDX-License-Identifier: GPL-2.0-or-later
#include "rend/metal4/metal4_presentation_timing.h"
#include <cstdio>
#include <array>
#include <cstdint>
#include <limits>
#include <deque>
#include <vector>

// Independent producer/callback events exercise the bounded pending-image queue.
// In particular, the producer can win the race at a display boundary. Deadlines
// that look regular in isolation must not evict still-valid queued contents.
static unsigned queueEvictions(double fieldHz, int gameInterval, double displayHz,
		double phase, bool previousExtraRefresh)
{
	struct Event { double at; bool producer; unsigned content; };
	std::vector<Event> events;
	const double sourceInterval = gameInterval / fieldHz, displayInterval = 1 / displayHz;
	for (unsigned n = 0; n < 600; ++n)
		events.push_back({n * sourceInterval + (n % 5) * .0004, true, n});
	for (unsigned n = 0; n * displayInterval < 600 * sourceInterval + 1; ++n)
		events.push_back({phase + n * displayInterval + (n % 3) * .0002, false, 0});
	std::stable_sort(events.begin(),events.end(),[](const Event& a,const Event& b) { return a.at < b.at; });
	metal::PresentationTimeline timeline;
	std::deque<double> queue;
	unsigned evicted = 0, presented = 0;
	for (const auto& event : events) {
		if (event.producer) {
			const double buffer = previousExtraRefresh ? 2 / fieldHz + displayInterval
				: metal::presentationBufferDuration(1 / fieldHz);
			const double target = timeline.videoFrame(100 + event.at,
				10 + event.content * sourceInterval, buffer, displayInterval);
			const unsigned capacity=previousExtraRefresh ? 3 : metal::presentationQueueCapacity;
			if (queue.size() == capacity) { queue.pop_front(); ++evicted; }
			queue.push_back(target);
		} else if (!queue.empty() && queue.front() <= 100 + event.at + displayInterval) {
			queue.pop_front(); ++presented;
		}
	}
	if (presented + evicted != 600 || !queue.empty()) return 600;
	return evicted;
}

struct CallbackSchedule {
	unsigned latePublications=0,evictions=0,presented=0,shortHolds=0,longHolds=0,maxPending=0;
	double minimumHeadroomMs=std::numeric_limits<double>::infinity();
};
// Independent callback/producer events model CAMetalDisplayLink's two future
// timestamps. A surface's presentation time can be several refreshes after
// its callback, while its CPU commit deadline is one refresh after callback.
// This checks publication/cadence only, not native GPU or display performance.
static CallbackSchedule callbackSchedule(unsigned fieldNumerator,unsigned fieldDenominator,
	unsigned gameInterval,unsigned displayHz,unsigned phase,unsigned advance,bool completeLead)
{
	constexpr unsigned frames=600;
	const double refresh=1./displayHz,field=double(fieldDenominator)/fieldNumerator;
	const double sourceStep=gameInterval*field,callbackPhase=double(phase)/displayHz/16;
	const double horizon=(advance+1)*refresh;
	const double lead=completeLead ? metal::displaySchedulingLead(100+refresh,100+horizon,displayHz)
		: advance*refresh;
	struct Event {double at;bool producer;unsigned content;};
	std::vector<Event> events;
	// The first earliest arrival establishes a stable initial phase. Later
	// arrivals exercise the whole +/-5 ms range without changing source time.
	constexpr int jitterMs[]{-5,5,2,-3,4,-1,3,-4,1,0};
	for(unsigned n=0;n<frames;++n)events.push_back({100+n*sourceStep+jitterMs[n%10]*.001,true,n});
	for(unsigned tick=0;tick<unsigned(std::ceil((frames*sourceStep+1)*displayHz));++tick)
		events.push_back({100+callbackPhase+tick*refresh,false,tick});
	std::stable_sort(events.begin(),events.end(),[](const auto& a,const auto& b){
		if(a.at!=b.at)return a.at<b.at;
		return a.producer && !b.producer;
	});
	metal::PresentationTimeline timeline;
	double oldOffset=0,oldPrevious=-1;
	struct Image {unsigned content;double target;};
	std::deque<Image> pending;
	std::vector<std::pair<unsigned,double>> presented;
	CallbackSchedule result;
	for(const auto& event:events) {
		if(event.producer) {
			const double source=10+event.content*sourceStep;
			double target;
			if(completeLead) target=timeline.videoFrame(event.at,source,
				metal::presentationBufferDuration(field),lead);
			else {
				// Historical one-field phase and partial callback lead are an
				// explicit negative control, unaffected by the adaptive timeline.
				target=oldOffset+source;
				if(oldPrevious<0 || source<oldPrevious || source-oldPrevious>1
						|| event.at-target>.1 || target-event.at>field+1.e-6)
					oldOffset=event.at+field-source;
				oldPrevious=source;target=oldOffset+source+lead;
			}
			// Find the first provided surface whose physical target is due.
			// A later publication cannot use that already delivered callback.
			const double tick=std::max(0.,std::ceil((target-horizon-100-callbackPhase)/refresh-1.e-9));
			const double callback=100+callbackPhase+tick*refresh;
			if(callback+1.e-9<event.at)++result.latePublications;
			result.minimumHeadroomMs=std::min(result.minimumHeadroomMs,(callback-event.at)*1000);
			const unsigned capacity=completeLead ? metal::presentationQueueCapacity : 3;
			if(pending.size()==capacity){pending.pop_front();++result.evictions;}
			pending.push_back({event.content,target});result.maxPending=std::max(result.maxPending,unsigned(pending.size()));
		} else {
			const double presentation=event.at+horizon;
			if(!pending.empty() && pending.front().target<=presentation+1.e-9) {
				presented.emplace_back(pending.front().content,presentation);pending.pop_front();
			}
		}
	}
	result.presented=unsigned(presented.size());
	for(size_t i=1;i<presented.size();++i) {
		const auto a=presented[i-1],b=presented[i];
		if(a.first<20 || b.first>=frames-1)continue;
		const auto ticks=uint64_t(std::llround((b.second-a.second)*displayHz));
		// Integer-rational oracle preserves PAL/NTSC and interval-2 fractions;
		// do not round each source image independently to two host refreshes.
		const uint64_t numerator=uint64_t(b.first-a.first)*gameInterval*fieldDenominator*displayHz;
		const uint64_t lower=numerator/fieldNumerator,upper=(numerator+fieldNumerator-1)/fieldNumerator;
		if(ticks<lower)++result.shortHolds;
		if(ticks>upper)++result.longHolds;
	}
	return result;
}

static bool callbackLookahead()
{
	for(double refresh:{60.,120.}) {
		const double expected=.010+1/refresh;
		if(std::abs(metal::displaySchedulingLead(100,100.010,refresh)-expected)>1.e-9
				|| std::abs(metal::displaySchedulingLead(100,100,refresh)-1/refresh)>1.e-9) {
			std::fputs("FAIL: Metal callback omitted the commit-deadline refresh from its presentation lookahead\n",stderr);return false;
		}
	}
	const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
	for(auto c:std::vector<std::array<double,3>>{{nan,100,60},{100,nan,60},{inf,100,60},{100,inf,60},
		{100,99,60},{100,100.010,0},{100,100.010,-1},{100,100.010,nan},{100,100.010,inf},{100,102,60},{100,100,1}})
		if(metal::displaySchedulingLead(c[0],c[1],c[2])!=0){std::fputs("FAIL: invalid Metal callback lookahead was accepted\n",stderr);return false;}
	unsigned cases=0,oldLate=0,oldShort=0,oldLong=0,maxPending=0;
	double minimumHeadroomMs=std::numeric_limits<double>::infinity();
	for(auto field:std::vector<std::pair<unsigned,unsigned>>{{50,1},{60000,1001},{60,1},{5979,100}})
	for(unsigned game:{1u,2u})for(unsigned display:{60u,120u})for(unsigned phase=0;phase<16;++phase)for(unsigned advance:{2u,4u}) {
		const auto old=callbackSchedule(field.first,field.second,game,display,phase,advance,false);
		const auto current=callbackSchedule(field.first,field.second,game,display,phase,advance,true);
		if(current.latePublications || current.evictions || current.shortHolds || current.longHolds || current.presented!=600 || current.maxPending>metal::presentationQueueCapacity) {
			std::fprintf(stderr,"FAIL: Metal callback lookahead field %u/%u interval %u display %u phase %u advance %u: late %u evicted %u short %u long %u presented %u pending %u\n",
				field.first,field.second,game,display,phase,advance,current.latePublications,current.evictions,current.shortHolds,current.longHolds,current.presented,current.maxPending);return false;
		}
		++cases;oldLate+=old.latePublications;oldShort+=old.shortHolds;oldLong+=old.longHolds;
		maxPending=std::max(maxPending,current.maxPending);minimumHeadroomMs=std::min(minimumHeadroomMs,current.minimumHeadroomMs);
	}
	if(!oldLate || !oldShort || !oldLong){std::fputs("FAIL: old partial lookahead did not reproduce late publication and uneven cadence\n",stderr);return false;}
	std::printf("PASS: %u Metal callback schedules, 600 images each, +/-5 ms producer jitter, integer-rational PAL/NTSC/59.79/interval-2 cadence; full lookahead has no late publications, evictions or avoidable short/long holds, pending <= %u, minimum callback headroom %.3f ms; old partial lead: %u late, %u short, %u long holds\n",
		cases,maxPending,minimumHeadroomMs,oldLate,oldShort,oldLong);
	return true;
}

struct AdaptiveSchedule {
	unsigned evicted=0,presented=0,maxPending=0,late=0,shortHolds=0,longHolds=0,corrections=0;
	double maximumLatencyMs=0;
	bool monotone=true,bounded=true,ordered=true;
};
// A render-cost phase changes twice while jitter spans up to 40 ms peak-to-peak.
// Arrivals are completed CPU submissions, not source-video timestamps; source
// time remains a separate exact rational stream. No GPU-readiness claim is made.
static AdaptiveSchedule adaptiveSchedule(unsigned numerator,unsigned denominator,unsigned game,
		unsigned display,unsigned phase,unsigned jitterMs,bool current,bool busy=false,bool burst=false)
{
	constexpr unsigned frames=1200;
	const double field=double(denominator)/numerator,step=game*field,refresh=1./display;
	const double advance=5*refresh,callbackPhase=phase*refresh/16;
	struct Event {double at;bool producer;unsigned index;};
	std::vector<Event> events;
	constexpr double jitter[]{-1,1,.4,-.6,.8,-.2,.6,-.8,.2,0};
	double previousArrival=0;
	for(unsigned n=0;n<frames;++n) {
		const double loadPhase=n>=400 && n<800 ? .014 : 0;
		double at=100+n*step+loadPhase+jitter[n%10]*jitterMs*.001;
		// A long stall followed by a deliberately excessive catch-up burst is
		// outside the bounded jitter guarantee. Overflow must stay explicit.
		if(burst && n>=400 && n<440)at=100+400*step+.250+(n-400)*.00001;
		at=std::max(previousArrival+.000001,at);previousArrival=at;
		events.push_back({at,true,n});
	}
	for(unsigned tick=0;tick<unsigned(std::ceil((frames*step+2)*display));++tick)
		events.push_back({100+callbackPhase+tick*refresh,false,tick});
	std::stable_sort(events.begin(),events.end(),[](const auto& a,const auto& b){
		return a.at!=b.at ? a.at<b.at : a.producer && !b.producer;
	});
	struct Image {unsigned index;double target,arrival;};
	std::deque<Image> pending;
	struct Output {unsigned index,tick;};
	std::vector<Output> output;
	metal::PresentationTimeline timeline;
	AdaptiveSchedule r;
	double oldOffset=0,lastTarget=0,lastSource=0;
	const unsigned capacity=current ? metal::presentationQueueCapacity : 4;
	for(const auto& e:events) {
		if(e.producer) {
			const double source=10+e.index*step;
			double target;
			if(current)target=timeline.videoFrame(e.at,source,field,advance);
			else {
				if(!e.index)oldOffset=e.at+field-source;
				// Negative control: a narrow upper clamp repeatedly undoes the
				// late-arrival correction when jitter exceeds its window.
				const double base=std::clamp(source+oldOffset,e.at+.5*field,e.at+2*field);
				oldOffset=base-source;target=base+advance;
			}
			if(e.index) {
				r.monotone &= target>=lastTarget;
				if(std::abs(target-lastTarget-(source-lastSource))>1.e-8)++r.corrections;
			}
			lastTarget=target;lastSource=source;
			const double reserve=target-advance-e.at;
			r.bounded &= reserve>=.5*field-1.e-8 && reserve<=3.5*field+1.e-8;
			const double intended=std::ceil((target-advance-100-callbackPhase)/refresh-1.e-9);
			if(100+callbackPhase+intended*refresh+1.e-9<e.at)++r.late;
			if(pending.size()==capacity){pending.pop_front();++r.evicted;}
			pending.push_back({e.index,target,e.at});r.maxPending=std::max(r.maxPending,unsigned(pending.size()));
		} else {
			if(busy && e.index==unsigned(600*step*display))continue;
			if(!pending.empty() && pending.front().target<=e.at+advance+1.e-9) {
				const auto image=pending.front();pending.pop_front();
				if(!output.empty())r.ordered &= output.back().index<image.index;
				output.push_back({image.index,e.index});
				r.maximumLatencyMs=std::max(r.maximumLatencyMs,(e.at+advance-image.arrival)*1000);
			}
		}
	}
	r.presented=unsigned(output.size());
	for(size_t n=1;n<output.size();++n) {
		const auto a=output[n-1],b=output[n];
		// Check natural cadence only after each phase settles, excluding the
		// explicit busy callback and its immediate FIFO recovery.
		auto stage=[](unsigned i){return i>20 && i<380 ? 0 : i>430 && i<780 ? 1 : i>830 && i<1180 ? 2 : -1;};
		if(stage(a.index)<0 || stage(a.index)!=stage(b.index) || (busy && a.index>=595 && a.index<=605))continue;
		const uint64_t ratio=uint64_t(b.index-a.index)*game*denominator*display;
		const uint64_t low=ratio/numerator,high=(ratio+numerator-1)/numerator;
		if(b.tick-a.tick<low)++r.shortHolds;
		if(b.tick-a.tick>high)++r.longHolds;
	}
	return r;
}

static bool adaptiveBuffering()
{
	// A 3.5-field horizon requires ceil(3.5)+1 regular queue slots at an
	// arrival/display boundary. GPU reads, visible image and producer leases
	// must fit independently; this does not promise to absorb unlimited bursts.
	if(metal::presentationQueueCapacity<unsigned(std::ceil(3.5))+1
			|| metal::presentationCompositionCapacity<metal::presentationQueueCapacity+1+2+1) {
		std::fputs("FAIL: adaptive queue/composition ownership budget is insufficient\n",stderr);return false;
	}
	unsigned cases=0,oldShort=0,oldLong=0,oldCorrections=0,maxPending=0;
	double maximumLatencyMs=0;
	for(auto field:std::vector<std::pair<unsigned,unsigned>>{{50,1},{5979,100},{60000,1001},{60,1}})
	for(unsigned game:{1u,2u})for(unsigned display:{60u,120u})for(unsigned jitter:{5u,20u})for(unsigned phase=0;phase<16;++phase) {
		const auto r=adaptiveSchedule(field.first,field.second,game,display,phase,jitter,true);
		if(r.evicted || r.late || r.presented!=1200 || !r.monotone || !r.bounded || !r.ordered
				|| r.shortHolds || r.longHolds || r.corrections>6 || r.maxPending>metal::presentationQueueCapacity) {
			std::fprintf(stderr,"FAIL: adaptive field%u/%u game%u display%u jitter%u phase%u: drops%u late%u shown%u short%u long%u corrections%u pending%u monotone%d bounded%d ordered%d\n",
				field.first,field.second,game,display,jitter,phase,r.evicted,r.late,r.presented,r.shortHolds,r.longHolds,r.corrections,r.maxPending,r.monotone,r.bounded,r.ordered);return false;
		}
		const auto old=adaptiveSchedule(field.first,field.second,game,display,phase,jitter,false);
		oldShort+=old.shortHolds;oldLong+=old.longHolds;oldCorrections+=old.corrections;
		maxPending=std::max(maxPending,r.maxPending);maximumLatencyMs=std::max(maximumLatencyMs,r.maximumLatencyMs);++cases;
	}
	if(!oldShort || !oldLong || oldCorrections<1000){std::fputs("FAIL: narrow adaptive window did not reproduce opposing phase corrections\n",stderr);return false;}
	for(unsigned display:{60u,120u})for(unsigned phase=0;phase<16;++phase) {
		const auto r=adaptiveSchedule(60,1,1,display,phase,20,true,true);
		if(r.evicted || r.presented!=1200 || !r.monotone || !r.ordered || r.maxPending>metal::presentationQueueCapacity) {
			std::fputs("FAIL: one busy callback exhausted adaptive queue reserve\n",stderr);return false;
		}
	}
	const auto burst=adaptiveSchedule(60,1,1,120,0,20,true,false,true);
	if(!burst.evicted || burst.presented+burst.evicted!=1200 || !burst.monotone || !burst.ordered
			|| burst.maxPending>metal::presentationQueueCapacity) {
		std::fputs("FAIL: excessive catch-up burst was hidden or broke bounded FIFO ordering\n",stderr);return false;
	}
	// PAL -> NTSC shrinks the upper horizon. Display lead may also change.
	// Newer timestamps can coincide during transition, but cannot go backward.
	metal::PresentationTimeline clock;
	clock.videoFrame(100,10,.020,.050);
	double last=clock.videoFrame(100.001,10.080,.020,.050);
	for(unsigned n=0;n<40;++n) {
		const double next=clock.videoFrame(100.002+n*.001,10.080+(n+1)/60.,1./60,n<20?.050:.020);
		if(next+1.e-9<last){std::fputs("FAIL: field/lead transition reversed queued targets\n",stderr);return false;}
		last=next;
	}
	const double settled=clock.videoFrame(100.5,11,1./60,.020);
	const double reserve=settled-.020-100.5;
	if(reserve<.5/60-1.e-9 || reserve>3.5/60+1.e-9){std::fputs("FAIL: field/lead transition did not settle back into the adaptive window\n",stderr);return false;}
	const double reset=clock.videoFrame(101,0,1./60,.020);
	if(std::abs(reset-(101+1./60+.020))>1.e-9){std::fputs("FAIL: source reset retained adaptive phase\n",stderr);return false;}
	const double backwardsHost=clock.videoFrame(50,1./60,1./60,.020);
	if(std::abs(backwardsHost-(50+1./60+.020))>1.e-9){std::fputs("FAIL: host clock reset retained an obsolete target\n",stderr);return false;}
	std::printf("PASS: %u adaptive schedules, 1200 images each, +/-5/20 ms arrivals, load phase +14/-14 ms, PAL/NTSC/59.79, interval1/2, 60/120Hz; no drops/late/cadence violations, pending<=%u, maximum modeled latency %.3f ms; narrow-window control %u short/%u long holds and %u corrections; busy callback reserve and explicit burst overflow verified\n",
		cases,maxPending,maximumLatencyMs,oldShort,oldLong,oldCorrections);
	return true;
}

int main()
{
	if(!callbackLookahead() || !adaptiveBuffering())return 1;
	struct Case { double hz; int game; bool gui, vsync, fast; double seconds; };
	// Ordinary frames inherit emulation cadence; no presentation duration
	// imposes a second cap, including after an interval-2 game opens the GUI.
	const Case cases[] = {
		{60, 1, false, true, false, 0}, {60, 2, false, true, false, 0},
		{120, 1, false, true, false, 0}, {120, 2, false, true, false, 0},
		{90, 1, false, true, false, 0}, {144, 1, false, true, false, 0},
		{90, 2, false, true, false, 0},
		{144, 2, false, true, false, 0}, {240, 1, false, true, false, 0},
		{50, 1, false, true, false, 0}, {59.94, 1, false, true, false, 0},
		{120, 2, true, true, false, 0}, {144, 2, true, true, false, 0},
		{120, 2, false, false, false, 0}, {120, 2, false, true, true, 0},
		{120, 2, true, true, true, 0}, {120, 0, false, true, false, 0},
		{120, 9, false, true, false, 0}, {0, 1, false, true, false, 0},
		{-1, 1, false, true, false, 0},
		{std::numeric_limits<double>::quiet_NaN(), 1, false, true, false, 0},
		{std::numeric_limits<double>::infinity(), 1, false, true, false, 0},
	};
	for (const auto& c : cases) {
		const double actual = metal::minimumFrameDuration(c.hz, c.game, c.gui, c.vsync, c.fast);
		if (std::abs(actual - c.seconds) > 1e-12) {
			std::fprintf(stderr, "FAIL: %.3f Hz, interval %d, GUI %d, VSync %d, fast %d: %.9f != %.9f\n", c.hz, c.game, c.gui, c.vsync, c.fast, actual, c.seconds);
			return 1;
		}
	}
	struct DuplicateCase { double hz; int game; bool gui, vsync, fast, duplicate; int copies; double seconds; };
	const DuplicateCase duplicateCases[] = {
		{120,1,false,true,false,true,2,1.0/120}, {120,2,false,true,false,true,4,1.0/120},
		{60,2,false,true,false,true,2,1.0/60}, {90,2,false,true,false,true,3,1.0/90},
		{144,2,false,true,false,true,4,1.0/144}, {240,2,false,true,false,true,8,1.0/240},
		{120,2,true,true,false,true,1,0}, {120,2,false,false,false,true,1,0},
		{120,2,false,true,true,true,1,0}, {120,2,false,true,false,false,1,0},
		{0,2,false,true,false,true,2,1.0/60},
		{std::numeric_limits<double>::infinity(),1,false,true,false,true,1,1.0/60},
	};
	for (const auto& c : duplicateCases) {
		if (metal::presentationCount(c.hz, c.game, c.gui, c.vsync, c.fast, c.duplicate) != c.copies
				|| std::abs(metal::minimumFrameDuration(c.hz, c.game, c.gui, c.vsync, c.fast, c.duplicate) - c.seconds) > 1e-12) {
			std::fprintf(stderr, "FAIL: duplicate policy %.3f Hz, interval %d\n", c.hz, c.game); return 1;
		}
	}
	// Independent integer-rational oracle: cumulative outputs must match the
	// display refreshes covered by the source fields, with less than one extra
	// refresh carried to the next source frame, even when intervals alternate.
	unsigned cadenceFrames=0;
	for(unsigned numerator:{50u,60u,60000u})for(unsigned display:{50u,60u,90u,100u,120u,144u,240u})
	for(unsigned mode:{1u,2u,3u}) {
		const unsigned denominator=numerator==60000?1001:1;
		metal::DuplicateCadence cadence;uint64_t accumulated=0,observed=0;
		for(unsigned n=0;n<10000;++n) {
			const unsigned interval=mode==3?1+n%2:mode;
			accumulated+=std::max(uint64_t(numerator),uint64_t(display)*interval*denominator);
			observed+=cadence.count(display,double(denominator)/numerator,interval,false,true,false,true);
			if(observed!=accumulated/numerator){std::fprintf(stderr,"FAIL: duplicate cadence field %u/%u display %u interval %u frame %u\n",numerator,denominator,display,interval,n);return 1;}
			++cadenceFrames;
		}
		for(unsigned disabled=0;disabled<4;++disabled){
			if(cadence.count(display,double(denominator)/numerator,2,disabled==0,disabled!=1,disabled==2,disabled!=3)!=1){std::fputs("FAIL: duplicate bypass mode\n",stderr);return 1;}
			metal::DuplicateCadence fresh;
			if(cadence.count(display,double(denominator)/numerator,2,false,true,false,true)!=fresh.count(display,double(denominator)/numerator,2,false,true,false,true)){std::fputs("FAIL: duplicate mode retained old phase\n",stderr);return 1;}
		}
	}
	{
		metal::DuplicateCadence cadence;
		for(const auto input:std::vector<std::pair<double,double>>{{120,1./50},{144,1./60},{120,1001./60000},{100,1./50}}){
			metal::DuplicateCadence fresh;
			for(unsigned n=0;n<19;++n)
				if(cadence.count(input.first,input.second,1,false,true,false,true)!=fresh.count(input.first,input.second,1,false,true,false,true)){
					std::fputs("FAIL: duplicate phase survived refresh/video-mode change\n",stderr);return 1;
				}
		}
		for(double field:{0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
			if(metal::presentationCount(120,2,false,true,false,true,field)!=4){std::fputs("FAIL: invalid video field fallback\n",stderr);return 1;}
	}
	// Rendering arrival jitter must not enter the source video timeline.
	for (double hz : {50.0, 59.94, 60.0}) {
		metal::PresentationTimeline clock;
		metal::PresentationTimeline displayClock;
		double source = 10, target = 0, lastSource = 0;
		for (int n = 0; n < 10000; ++n) {
			const double arrival = 100 + source - 10 + (n % 5 - 2) * 0.002;
			const double next = clock.videoFrame(arrival, source, 2 / hz + 1.0 / 120);
			// The display can request a composition well before it appears.
			// Its lead must shift presentation, without shrinking the producer's
			// jitter buffer or changing the emulated video cadence.
			const double lead = n < 5000 ? 5.0 / 120 : 3.0 / 60;
			const double display = displayClock.videoFrame(arrival, source, 2 / hz + 1.0 / 120, lead);
			if (std::abs(display - next - lead) > 1e-9) {
				std::fprintf(stderr, "FAIL: display lead changed the source clock at %.2f Hz\n", hz); return 1;
			}
			if (n && std::abs((next - target) - (source - lastSource)) > 1e-9) {
				std::fprintf(stderr, "FAIL: video clock drift at %.2f Hz, frame %d\n", hz, n); return 1;
			}
			target = next; lastSource = source;
			// Arbitrary game cadence, including alternating intervals.
			source += (1 + n % 3) / hz;
		}
		if (clock.videoFrame(1000, source, .05) < 1000
				|| clock.videoFrame(1001, 0, .05) < 1001
				|| clock.videoFrame(1002, std::numeric_limits<double>::quiet_NaN(), .05) != 0) {
			std::fprintf(stderr, "FAIL: pause/reset/invalid clock handling\n"); return 1;
		}
	}
	unsigned previousEvictions = 0;
	for (double fieldHz : {50.0, 59.94, 60.0})
		for (int interval : {1,2})
			for (double displayHz : {60.0,120.0})
				for (unsigned phase = 0; phase < 16; ++phase) {
					if (queueEvictions(fieldHz,interval,displayHz,phase / displayHz / 16,false)) {
						std::fprintf(stderr,"FAIL: valid contents evicted at field %.2f interval %d display %.0f phase %u\n",
							fieldHz,interval,displayHz,phase); return 1;
					}
					previousEvictions += queueEvictions(fieldHz,interval,displayHz,phase / displayHz / 16,true);
				}
	if (!previousEvictions) { std::fprintf(stderr,"FAIL: queue regression did not detect the previous overlong horizon\n"); return 1; }
	std::printf("PASS: 34 presentation policies, %u integer-oracle duplicate frames, 60000 PAL/NTSC deadlines and 192 producer/display queue schedules; previous horizon evicted %u valid contents\n",cadenceFrames,previousEvictions);
	return 0;
}
