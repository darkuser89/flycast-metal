// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace metal {
// Five regular pending images cover the adaptive horizon and its boundary;
// one extra slot absorbs a busy display callback. GPU replay stays two-deep.
inline constexpr unsigned presentationQueueCapacity = 6;
inline constexpr unsigned presentationCompositionCapacity = presentationQueueCapacity + 4;
// Budget the full nominal callback-to-screen advance: one display period
// before the CPU submission deadline, plus the deadline-to-screen lead.
// Omitting that period consumes it from the producer's video-field buffer.
inline double displaySchedulingLead(double deadline, double presentation, double refreshRate)
{
	if (!std::isfinite(deadline) || !std::isfinite(presentation) || !std::isfinite(refreshRate)
			|| refreshRate <= 0 || presentation < deadline) return 0;
	const double lead = presentation - deadline + 1 / refreshRate;
	return std::isfinite(lead) && lead > 0 && lead < 1 ? lead : 0;
}
// Start with one video field. PresentationTimeline can retain more of the
// producer's arrival variation without changing the emulated source clock.
inline double presentationBufferDuration(double videoFieldDuration)
{
	return videoFieldDuration;
}
// Keep a common host-time deadline across submissions. GPU/CPU delays must
// not be added to every subsequent frame or duplicate presentation.
class PresentationTimeline {
public:
	void reset() { source = 0; host = 0; previous = -1; previousHost = 0; previousTarget = 0; }
	// Source time is captured at the emulated video swap that produced the
	// image, retaining sub-field precision. No assumed 30/60 Hz game rate and no accumulated host jitter.
	double videoFrame(double now, double videoTime, double latency, double displayLead = 0) {
		if (!std::isfinite(now) || !std::isfinite(videoTime) || !std::isfinite(latency)
				|| !std::isfinite(displayLead) || displayLead < 0 || latency <= 0) { reset(); return 0; }
		double target = host + videoTime - source;
		const bool restart = previous < 0 || videoTime < previous || videoTime - previous > 1
				|| now < previousHost;
		if (restart) {
			source = videoTime; host = now + latency; target = host;
		} else {
			// A heavier render phase can exhaust the initial field of buffering.
			// Move the host phase only by the missing margin, not by a full field
			// on every late arrival. A wider upper boundary supplies hysteresis;
			// early arrivals must not immediately undo the late-arrival correction.
			// Outside this bounded horizon, catch-up bursts may still overflow the
			// display queue. Never turn them into unlimited latency or CPU waits.
			double bounded = std::clamp(target, now + latency * .5, now + latency * 3.5);
			// A field-mode or display-lead change must not timestamp a newer
			// queued image before an older one. Drain that short transition in FIFO order.
			bounded = std::max(bounded, previousTarget - displayLead);
			if (bounded != target) { source = videoTime; host = bounded; target = bounded; }
		}
		previous = videoTime; previousHost = now;
		// Producer buffering and the display's advance request are different
		// intervals. Only producer buffering bounds the pending image queue.
		previousTarget = target + displayLead;
		return previousTarget;
	}
private:
	double source = 0, host = 0, previous = -1, previousHost = 0, previousTarget = 0;
};
// Explicit repeats follow the emulated video field, including PAL/NTSC and
// interval-2 games. Carry fractional refreshes across source frames instead of
// rounding each field before applying the game's interval.
class DuplicateCadence {
public:
	void reset(){fraction=0;refresh=field=0;}
	int count(double refreshRate,double videoFieldDuration,int gameInterval,
		bool guiOpen,bool vsync,bool fastForward,bool duplicateFrames)
	{
		if(!duplicateFrames || guiOpen || !vsync || fastForward){reset();return 1;}
		if(!std::isfinite(refreshRate) || refreshRate<=0)refreshRate=60;
		if(!std::isfinite(videoFieldDuration) || videoFieldDuration<=0)videoFieldDuration=1./60;
		if(refresh!=refreshRate || field!=videoFieldDuration){reset();refresh=refreshRate;field=videoFieldDuration;}
		const double maximum=double(std::numeric_limits<int>::max());
		double duration=refreshRate*videoFieldDuration*std::clamp(gameInterval,1,2);
		if(!std::isfinite(duration))duration=maximum;
		// Duplication never discards a source frame on a slower display.
		const double total=fraction+std::clamp(duration,1.0,maximum);
		const int copies=int(std::min(maximum,std::floor(total+1e-9)));
		fraction=std::max(0.0,total-copies);
		if(fraction>=1)fraction=0; // Saturated integer range, not a valid display ratio.
		return copies;
	}
private:
	double fraction=0,refresh=0,field=0;
};
// First-frame query for callers which do not own a continuing source stream.
inline int presentationCount(double refreshRate, int gameInterval, bool guiOpen, bool vsync, bool fastForward,
	bool duplicateFrames,double videoFieldDuration=1./60)
{
	DuplicateCadence cadence;
	return cadence.count(refreshRate,videoFieldDuration,gameInterval,guiOpen,vsync,fastForward,duplicateFrames);
}
inline double minimumFrameDuration(double refreshRate, int /*gameInterval*/, bool guiOpen, bool vsync, bool fastForward, bool duplicateFrames = false)
{
	// Ordinary presents and GUI frames have no second duration limit.
	// Reserve timed presentation for explicitly requested copies.
	if (!duplicateFrames || guiOpen || !vsync || fastForward) return 0;
	if (!std::isfinite(refreshRate) || refreshRate <= 0) refreshRate = 60;
	return 1 / refreshRate;
}
}
