// SPDX-License-Identifier: GPL-2.0-or-later
// Standalone ViSP adapter. Frames arrive as raw grayscale bytes on stdin;
// pose estimates are written to a separate CSV, never mixed into that stream.
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpImage.h>
#include <visp3/mbt/vpMbGenericTracker.h>
#include <visp3/me/vpMe.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

int main(int argc, char** argv) {
  if (argc != 4) { std::cerr << "tracker model.cao seed.txt output.csv\n"; return 2; }
  std::ifstream seed(argv[2]);
  unsigned width, height; double fx, fy, cx, cy;
  seed >> width >> height >> fx >> fy >> cx >> cy;
  vpHomogeneousMatrix cMo;
  for (int i=0;i<4;++i) for (int j=0;j<4;++j) seed >> cMo[i][j];
  if (!seed) { std::cerr << "Invalid seed\n"; return 2; }
  vpImage<unsigned char> image(height,width);
  vpCameraParameters camera; camera.initPersProjWithoutDistortion(fx,fy,cx,cy);
  vpMbGenericTracker tracker;
  tracker.setTrackerType(vpMbGenericTracker::EDGE_TRACKER);
  tracker.setCameraParameters(camera);
  vpMe me; me.setMaskSize(5); me.setMaskNumber(180); me.setRange(10);
  me.setLikelihoodThresholdType(vpMe::NORMALIZED_THRESHOLD);
  me.setThreshold(15); me.setMu1(.5); me.setMu2(.5); me.setSampleStep(3);
  tracker.setMovingEdge(me);
  tracker.setProjectionErrorComputation(true);
  tracker.setGoodMovingEdgesRatioThreshold(.35);
  tracker.setAngleAppear(vpMath::rad(65)); tracker.setAngleDisappear(vpMath::rad(75));
  tracker.loadModel(argv[1],false);
  std::ofstream output(argv[3]); output << std::setprecision(12);
  output << "frame,tracker_ok,projection_error_deg,edge_features,milliseconds";
  for(int i=0;i<16;++i) output << ",m" << i;
  output << "\n";
  bool initialized=false; unsigned frame=0;
  while(std::cin.read(reinterpret_cast<char*>(image.bitmap), width*height)) {
    bool ok=true; double error=90; unsigned features=0;
    auto start=std::chrono::steady_clock::now();
    try {
      if(!initialized) { tracker.initFromPose(image,cMo); initialized=true; }
      else tracker.track(image);
      tracker.getPose(cMo);
      error=tracker.getProjectionError(); features=tracker.getNbFeaturesEdge();
    } catch(const vpException& e) {
      ok=false;
      if(frame%60==0) std::cerr << "Tracking exception at " << frame << ": " << e.what() << "\n";
    }
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    output << frame << ',' << ok << ',' << error << ',' << features << ',' << ms;
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) output << ',' << cMo[i][j];
    output << '\n';
    ++frame;
  }
  std::cerr << "Processed " << frame << " frames\n";
}
