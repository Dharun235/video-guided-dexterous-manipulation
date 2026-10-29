// SPDX-License-Identifier: MIT
// Headless, offline M3T adapter. Input: BGR frames on stdin, known mesh, one pose.
#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/region_model.h>
#include <m3t/region_modality.h>
#include <m3t/renderer_geometry.h>
#include <m3t/tracker.h>
#include <m3t/link.h>
#include <m3t/optimizer.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

class PipeCamera : public m3t::ColorCamera {
 public:
  PipeCamera(int w,int h,float fx,float fy,float cx,float cy) : ColorCamera("pipe_camera") {
    intrinsics_={fx,fy,cx,cy,w,h}; image_=cv::Mat(h,w,CV_8UC3,cv::Scalar(0,0,0));
  }
  bool SetUp() override { set_up_=true; return true; }
  bool UpdateImage(bool) override {
    return bool(std::cin.read(reinterpret_cast<char*>(image_.data),image_.total()*3));
  }
};

int main(int argc,char**argv) {
  if(argc!=4) { std::cerr<<"tracker mesh.obj seed.txt output.csv\n"; return 2; }
  std::ifstream seed(argv[2]); int w,h;float fx,fy,cx,cy;seed>>w>>h>>fx>>fy>>cx>>cy;
  m3t::Transform3fA pose=m3t::Transform3fA::Identity();
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)seed>>pose.matrix()(i,j);
  if(!seed)return 2;
  auto camera=std::make_shared<PipeCamera>(w,h,fx,fy,cx,cy);
  auto body=std::make_shared<m3t::Body>("case",argv[1],1.f,true,true,m3t::Transform3fA::Identity());
  auto model=std::make_shared<m3t::RegionModel>("case_region",body,
      std::filesystem::path(argv[3]).parent_path()/"region_model.bin",.3f,2,150,.03f,.002f,false,512);
  auto modality=std::make_shared<m3t::RegionModality>("case_region",body,camera,model);
  modality->set_n_lines_max(150);
  auto link=std::make_shared<m3t::Link>("case_link",body);link->AddModality(modality);
  auto optimizer=std::make_shared<m3t::Optimizer>("case_optimizer",link);
  m3t::Tracker tracker("case_tracker",5,2);
  tracker.AddOptimizer(optimizer);
  if(!tracker.SetUp()) {std::cerr<<"SETUP_FAILED\n";return 3;}
  body->set_body2world_pose(pose);
  link->set_link2world_pose(pose);
  tracker.StartTracking();
  std::ofstream out(argv[3]);out<<std::setprecision(12);
  out<<"frame,tracker_ok,projection_error_deg,edge_features,milliseconds";
  for(int i=0;i<16;++i)out<<",m"<<i;out<<"\n";
  int frame=0;
  while(tracker.UpdateCameras(frame)) {
    auto start=std::chrono::steady_clock::now();
    bool ok=tracker.ExecuteStartingStep(frame)&&tracker.ExecuteTrackingStep(frame);
    pose=body->body2world_pose();
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    // Edge-specific metrics are intentionally unavailable for this region tracker.
    out<<frame<<','<<ok<<",nan,-1,"<<ms;
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)out<<','<<pose.matrix()(i,j);
    out<<'\n';++frame;
  }
  std::cerr<<"Processed "<<frame<<" frames\n";
}
