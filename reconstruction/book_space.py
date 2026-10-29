"""Fit a photographic tabletop scene from book5; export separate collision physics.

This is constrained video-guided reconstruction, not dense neural reconstruction.
One estimated paperback dimension fixes the otherwise ambiguous metric scale.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.optimize import minimize_scalar, least_squares
from scipy.spatial.transform import Rotation


BOOK_QUAD = np.float32([[222, 199], [498, 201], [502, 374], [187, 367]])
BOOK_SIZE = np.array([.198, .129, .018])
CASE_SIZE = np.array([.065, .048, .026])
SOURCE_SHA256 = "fd2201d3349b5184f9d59959c33e7fa5de283e43ec9786b0620636fef6ad702c"


def corners(z):
    x, y = BOOK_SIZE[:2] / 2
    return np.float64([[-x, y, z], [x, y, z], [x, -y, z], [-x, -y, z]])


def project(xyz, k, rot, trans):
    return cv2.projectPoints(np.float64(xyz), rot, trans, k, None)[0].reshape(-1, 2)


def fit_camera(quad, focal=None):
    def solve(f):
        k = np.array([[f, 0, 416.], [0, f, 232.], [0, 0, 1.]])
        _, r, t = cv2.solvePnP(corners(BOOK_SIZE[2]), np.float64(quad), k, None)
        error = np.linalg.norm(project(corners(BOOK_SIZE[2]), k, r, t) - quad, axis=1)
        return k, r, t, error
    if focal is None:
        focal = minimize_scalar(lambda f: np.mean(solve(f)[3]**2),
                                bounds=(400., 1600.), method="bounded").x
    return solve(focal)


def rays_to_plane(pixels, z, k, r, t):
    rot = cv2.Rodrigues(r)[0]
    center = (-rot.T @ t).ravel()
    dirs = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(k).T @ rot
    distance = (z - center[2]) / dirs[:, 2]
    return center + distance[:, None] * dirs


def textured_quad(output, name, xyz, photo, k, r, t, mask=None, resolution=(1024, 768)):
    uv = np.float32([[0, 0], [resolution[0]-1, 0],
                     [resolution[0]-1, resolution[1]-1], [0, resolution[1]-1]])
    image = photo if mask is None else cv2.inpaint(photo, mask, 9, cv2.INPAINT_TELEA)
    src = project(xyz, k, r, t).astype(np.float32)
    h = cv2.getPerspectiveTransform(src, uv)
    tex = cv2.warpPerspective(image, h, resolution, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(181, 192, 201))
    cv2.imwrite(str(output / f"{name}.png"), tex)
    with (output / f"{name}.obj").open("w") as f:
        for x in xyz:
            f.write("v " + " ".join(map(str, x)) + "\n")
        for u, v in [(0, 1), (1, 1), (1, 0), (0, 0)]:
            f.write(f"vt {u} {v}\n")
        f.write("f 1/1 4/4 3/3\nf 1/1 3/3 2/2\n")


def case_vertices(size):
    w, d, h = size
    radius = min(.011, d*.23)
    xy = []
    for cx, cy, start in [(w/2-radius, d/2-radius, 0),
                          (-w/2+radius, d/2-radius, 90),
                          (-w/2+radius, -d/2+radius, 180),
                          (w/2-radius, -d/2+radius, 270)]:
        for angle in np.linspace(start, start+90, 12, endpoint=False):
            a = np.deg2rad(angle)
            xy.append([cx+radius*np.cos(a), cy+radius*np.sin(a)])
    xy = np.array(xy)
    vertices = np.vstack([np.c_[xy*s, np.full(len(xy), z)]
                          for s, z in [(.87, -h/2), (1., -h/2+.004),
                                       (1., h/2-.004), (.87, h/2)]])
    return vertices


def rounded_case(output):
    import trimesh
    mesh = trimesh.convex.convex_hull(case_vertices(CASE_SIZE))
    mesh.export(output / "case.obj")


def build(video, output):
    if hashlib.sha256(video.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("This scene's reviewed annotations apply only to the original book5.mp4")
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    cap = cv2.VideoCapture(str(video))
    ok, photo = cap.read()
    if not ok or photo.shape[:2] != (464, 832):
        raise ValueError("Reviewed annotations apply only to the 832x464 book5 clip")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    k, r, t, error = fit_camera(BOOK_QUAD)
    rotation = cv2.Rodrigues(r)[0]
    camera_pos = (-rotation.T @ t).ravel()
    camera_rot = rotation.T @ np.diag([1, -1, -1])
    camera_quat_xyzw = Rotation.from_matrix(camera_rot).as_quat()
    camera_quat = camera_quat_xyzw[[3, 0, 1, 2]]
    # Fit the case's visible silhouette, while keeping thickness as a stated prior.
    initial=rays_to_plane(np.float32([[356,273]]),BOOK_SIZE[2]+CASE_SIZE[2]/2,k,r,t)[0]
    crop=np.zeros(photo.shape[:2],np.uint8);crop[230:313,300:415]=255
    dark=((cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)<95)&(crop>0)).astype(np.uint8)*255
    contour=max(cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea)[:,0,:]
    theta=np.linspace(0,2*np.pi,48,endpoint=False)
    directions=np.c_[np.cos(theta),np.sin(theta)]
    target_support=(contour@directions.T).max(0)
    def case_residual(p):
        yaw=Rotation.from_euler('z',p[4]).as_matrix()
        vertices=case_vertices([p[0],p[1],CASE_SIZE[2]])@yaw.T+[p[2],p[3],initial[2]]
        projected=project(vertices,k,r,t)
        return (projected@directions.T).max(0)-target_support
    fit=least_squares(case_residual,[*CASE_SIZE[:2],*initial[:2],0.],
                      bounds=([.035,.025,initial[0]-.03,initial[1]-.03,-.8],
                              [.09,.08,initial[0]+.03,initial[1]+.03,.8]))
    CASE_SIZE[:2]=fit.x[:2]
    case_pos=np.array([fit.x[2],fit.x[3],initial[2]])
    case_yaw=fit.x[4]
    case_rotation=Rotation.from_euler('z',case_yaw).as_matrix()
    # The rear shelf edge is visible; the near edge is outside the image.
    table_pixels = np.float32([[0, 150], [831, 155], [831, 520], [0, 520]])
    table = rays_to_plane(table_pixels, 0., k, r, t)
    # Static photographic appearance is separate from closed collision boxes.
    exclusion = np.zeros(photo.shape[:2], np.uint8)
    cv2.fillConvexPoly(exclusion, BOOK_QUAD.astype(int), 255)
    exclusion = cv2.dilate(exclusion, np.ones((61, 61), np.uint8))
    textured_quad(assets, "table", table + [0, 0, .0002], photo, k, r, t, exclusion)
    textured_quad(assets, "book", corners(BOOK_SIZE[2]+.0003), photo, k, r, t,
                  mask=cv2.rectangle(np.zeros_like(exclusion), (298, 226), (415, 315), 255, -1),
                  resolution=(1024, 668))
    # The last frame exposes the actual book cover previously hidden by the case.
    cap.set(cv2.CAP_PROP_POS_FRAMES,count-1);last_ok,last_photo=cap.read()
    if last_ok:
        sift0=cv2.SIFT_create(nfeatures=2000)
        static=np.zeros(photo.shape[:2],np.uint8);cv2.fillConvexPoly(static,BOOK_QUAD.astype(int),255)
        static[220:325,290:425]=0
        a,da=sift0.detectAndCompute(photo,static);b,db=sift0.detectAndCompute(last_photo,None)
        good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.7*n.distance]
        aa=np.float32([a[m.queryIdx].pt for m in good]);bb=np.float32([b[m.trainIdx].pt for m in good])
        hom,inliers=cv2.findHomography(aa,bb,cv2.RANSAC,3)
        if hom is not None and inliers.sum()>=12:
            last_quad=cv2.perspectiveTransform(BOOK_QUAD[None],hom)[0]
            uv=np.float32([[0,0],[1023,0],[1023,667],[0,667]])
            tex=cv2.warpPerspective(last_photo,cv2.getPerspectiveTransform(last_quad,uv),(1024,668))
            cv2.imwrite(str(assets/'book.png'),tex)
    wall = np.array([table[0]+[0, 0, .32], table[1]+[0, 0, .32], table[1], table[0]])
    textured_quad(assets, "wall", wall, photo, k, r, t)
    rounded_case(assets)
    tx,ty=CASE_SIZE[:2]*.43
    top = np.array([[-tx,ty,.0132],[tx,ty,.0132],[tx,-ty,.0132],[-tx,-ty,.0132]])
    textured_quad(assets, "case_top", top@case_rotation.T+case_pos, photo, k, r, t,
                  resolution=(512, 384))
    # Store the case-top mesh in its moving body's local coordinate frame.
    text=(assets/'case_top.obj').read_text().splitlines()
    (assets/'case_top.obj').write_text('\n'.join(
        'v '+' '.join(map(str,(np.array(line.split()[1:],float)-case_pos)@case_rotation))
        if line.startswith('v ') else line for line in text)+'\n')
    root = ET.Element("mujoco", model="book5_video_fitted_tabletop")
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets", texturedir="assets")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    visual=ET.SubElement(root,"visual")
    ET.SubElement(visual,"global",offwidth="1280",offheight="720")
    ET.SubElement(visual,"headlight",ambient="0.55 0.55 0.55",diffuse="0.45 0.45 0.45",specular="0 0 0")
    default=ET.SubElement(root,"default")
    ET.SubElement(default,"geom",friction="0.7 0.005 0.001",solref="0.005 1",solimp="0.95 0.99 0.001")
    asset=ET.SubElement(root,"asset")
    for name in ['table','book','wall','case_top']:
        ET.SubElement(asset,"texture",name=name,type="2d",file=f"{name}.png")
        ET.SubElement(asset,"material",name=name,texture=name,texuniform="false",specular="0",shininess="0",reflectance="0")
        ET.SubElement(asset,"mesh",name=name,file=f"{name}.obj",inertia="shell")
    ET.SubElement(asset,"mesh",name="case",file="case.obj")
    world=ET.SubElement(root,"worldbody")
    fmt=lambda x:' '.join(f'{v:.9g}' for v in x)
    ET.SubElement(world,"camera",name="source",pos=fmt(camera_pos),quat=fmt(camera_quat),
                  fovy=str(np.rad2deg(2*np.arctan(232/k[1,1]))))
    ET.SubElement(world,"light",pos="0 -.2 1",dir="0 0 -1",diffuse=".35 .35 .35",castshadow="false")
    for name in ['table','book','wall']:
        ET.SubElement(world,"geom",name=name+'_appearance',type="mesh",mesh=name,material=name,
                      contype="0",conaffinity="0",group="1")
    # A broad shelf slab is an estimated completion outside the photographed patch.
    low,high=table[:,:2].min(0),table[:,:2].max(0)
    ET.SubElement(world,"geom",name="table_collision",type="box",pos=fmt([*(low+high)/2,-.015]),
                  size=fmt([*(high-low)/2,.015]),rgba=".78 .74 .65 0",group="3")
    ET.SubElement(world,"geom",name="book_collision",type="box",pos=fmt([0,0,BOOK_SIZE[2]/2]),
                  size=fmt(BOOK_SIZE/2),rgba=".72 .66 .51 1")
    wall_mid=(table[0]+table[1])/2+[0,.006,.16]
    wall_angle=np.arctan2(table[1,1]-table[0,1],table[1,0]-table[0,0])
    ET.SubElement(world,"geom",name="wall_collision",type="box",pos=fmt(wall_mid),
                  size=fmt([np.linalg.norm(table[1]-table[0])/2,.006,.16]),
                  euler=fmt([0,0,wall_angle]),rgba=".55 .55 .55 0",group="3")
    body=ET.SubElement(world,"body",name="case",pos=fmt(case_pos),quat=fmt([np.cos(case_yaw/2),0,0,np.sin(case_yaw/2)]))
    ET.SubElement(body,"freejoint",name="case_free")
    ET.SubElement(body,"geom",name="case_solid",type="mesh",mesh="case",mass=".055",rgba=".035 .04 .035 1")
    ET.SubElement(body,"geom",name="case_top_appearance",type="mesh",mesh="case_top",material="case_top",
                  contype="0",conaffinity="0",mass="0",group="1")
    ET.indent(root)
    ET.ElementTree(root).write(output/'scene.xml',encoding="unicode")
    # Static feature registration measures camera movement without inventing depth.
    sift=cv2.SIFT_create(nfeatures=2500)
    mask=np.zeros(photo.shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,BOOK_QUAD.astype(int),255)
    mask[220:325,290:425]=0
    kp0,des0=sift.detectAndCompute(photo,mask)
    tracks=[];overlays=[];last=np.array([356.,273.])
    for index in np.unique(np.linspace(0,count-1,26).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,im=cap.read()
        if not ok:continue
        kp,des=sift.detectAndCompute(im,None)
        matches=cv2.BFMatcher().knnMatch(des0,des,k=2) if des is not None else []
        good=[m for m,n in matches if m.distance<.7*n.distance]
        cam=None
        if len(good)>=8:
            a=np.float32([kp0[m.queryIdx].pt for m in good]);b=np.float32([kp[m.trainIdx].pt for m in good])
            hom,inliers=cv2.findHomography(a,b,cv2.RANSAC,3)
            if hom is not None and inliers.sum()>=8:
                q=cv2.perspectiveTransform(BOOK_QUAD[None],hom)[0]
                kk,rr,tt,ee=fit_camera(q,k[0,0])
                cam=dict(inliers=int(inliers.sum()),book_quad=q.tolist(),rotation=rr.ravel().tolist(),
                         translation=tt.ravel().tolist(),corner_fit_px=float(ee.mean()))
                cv2.polylines(im,[q.astype(int)],True,(0,255,255),2)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        dark=((hsv[:,:,2]<100)&(hsv[:,:,1]<115)).astype(np.uint8)*255
        dark=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        contours,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for contour in contours:
            area=cv2.contourArea(contour);x,y,w,h=cv2.boundingRect(contour)
            if 900<area<16000 and 25<w<170 and 20<h<160 and y>150:
                center=np.array([x+w/2,y+h/2])
                candidates.append((np.linalg.norm(center-last),contour,center))
        obj=None
        if candidates:
            _,contour,center=min(candidates,key=lambda x:x[0]);last=center
            solidity=cv2.contourArea(contour)/max(cv2.contourArea(cv2.convexHull(contour)),1)
            if solidity>=.72:
                cv2.drawContours(im,[contour],-1,(255,180,0),2)
                obj=dict(center_px=center.tolist(),visible_mask_area_px=float(cv2.contourArea(contour)),
                         contour=contour[:,0,:].tolist(),solidity=float(solidity),depth="unresolved during lifting")
        tracks.append(dict(source_frame=int(index),seconds=float(index/fps),camera=cam,case=obj))
        cv2.putText(im,f'{index/fps:.2f}s | yellow: book registration | blue: case mask',
                    (8,24),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
        overlays.append(cv2.resize(im,(416,232)))
    cap.release()
    while len(overlays)%4:overlays.append(np.zeros_like(overlays[0]))
    cv2.imwrite(str(output/'tracking-sheet.jpg'),np.vstack([np.hstack(overlays[i:i+4]) for i in range(0,len(overlays),4)]))
    (output/'tracks.json').write_text(json.dumps(tracks,indent=2))
    report=dict(source=str(video.resolve()),source_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                method="Photographic plane fitting + explicit rounded rigid case; no neural scene mesh claimed",
                camera=dict(intrinsics=k.tolist(),rotation=r.ravel().tolist(),translation=t.ravel().tolist(),
                            position=camera_pos.tolist(),book_corner_mean_fit_px=float(error.mean())),
                priors=dict(book_dimensions_m=BOOK_SIZE.tolist(),case_dimensions_m=CASE_SIZE.tolist(),
                            case_mass_kg=.055,friction=.7,all="estimates, not measured from the video"),
                initial_case_center_m=case_pos.tolist(),table_polygon_m=table.tolist(),
                case_fit=dict(yaw_radians=float(case_yaw),silhouette_support_rmse_px=float(np.sqrt(np.mean(fit.fun**2))),
                              note="Fit to visible silhouette conditional on estimated height and camera calibration"),
                sampled_frames=len(tracks),registered_frames=sum(x['camera'] is not None for x in tracks),
                segmented_case_frames=sum(x['case'] is not None for x in tracks),
                limitations=["Metric scale fixed by estimated paperback dimensions",
                             "Only visible tabletop and rear wall are reconstructed",
                             "Hidden surfaces and slab thickness are authored completions",
                             "Photographic textures contain baked lighting and inpainted occlusions",
                             "Case mask is visible silhouette only; full 6D moving pose not recovered",
                             "Physics parameters are estimates; no robot grasp attempted"])
    (output/'reconstruction.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def verify(output, render=False):
    import mujoco
    model=mujoco.MjModel.from_xml_path(str((output/'scene.xml').resolve()))
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    case=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'case')
    initial=data.xpos[case].copy()
    checks=[]
    for label,xy,z,expected in [('on_book',initial[:2],.14,BOOK_SIZE[2]+CASE_SIZE[2]/2),
                               ('on_table',[.18,-.04],.14,CASE_SIZE[2]/2)]:
        mujoco.mj_resetData(model,data)
        data.qpos[:3]=[*xy,z]
        max_pen=0.;peak_force=0.
        for step in range(1500):
            mujoco.mj_step(model,data)
            for c in data.contact:
                max_pen=max(max_pen,-c.dist)
            if step%10==0:
                for j in range(data.ncon):
                    force=np.zeros(6);mujoco.mj_contactForce(model,data,j,force)
                    peak_force=max(peak_force,float(force[0]))
        checks.append(dict(name=label,final_z=float(data.xpos[case,2]),expected_z=float(expected),
                           speed=float(np.linalg.norm(data.qvel)),max_penetration_m=float(max_pen),
                           peak_contact_force_n=peak_force,passed=bool(abs(data.xpos[case,2]-expected)<.005
                           and np.linalg.norm(data.qvel)<.01 and max_pen<.006)))
    result=dict(finite_physics=bool(np.isfinite(data.qpos).all()),checks=checks,
                passed=all(c['passed'] for c in checks),scope="Environment gravity/contact checks, not manipulation")
    (output/'physics-report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if render:
        mujoco.mj_resetData(model,data);mujoco.mj_forward(model,data)
        renderer=mujoco.Renderer(model,height=464,width=832)
        renderer.update_scene(data,camera='source')
        cv2.imwrite(str(output/'source-view.png'),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
        cam=mujoco.MjvCamera();cam.lookat[:]=[.06,0,0];cam.distance=.70;cam.azimuth=120;cam.elevation=-52
        renderer.update_scene(data,camera=cam)
        cv2.imwrite(str(output/'overview.png'),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
        cap=cv2.VideoCapture(str(Path(json.loads((output/'reconstruction.json').read_text())['source'])))
        _,photo=cap.read();cap.release()
        preview=cv2.imread(str(output/'source-view.png'))
        cv2.imwrite(str(output/'comparison.jpg'),np.hstack([photo,preview]))
        renderer.close()


def record(output):
    import mujoco
    import subprocess
    model=mujoco.MjModel.from_xml_path(str((output/'scene.xml').resolve()))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,height=464,width=832)
    writer=cv2.VideoWriter(str(output/'preview-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(832,464))
    cam=mujoco.MjvCamera();cam.lookat[:]=[.04,.01,.025];cam.distance=.7
    for i in range(120):
        cam.azimuth=85+i*.7;cam.elevation=-48
        renderer.update_scene(data,camera=cam)
        image=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        cv2.putText(image,'Video-fitted scene | estimated dimensions and physics',(12,28),0,.55,(255,255,255),1)
        writer.write(image)
    for label,xy in [('Drop onto book',[0,0]),('Drop onto table',[.18,-.04])]:
        mujoco.mj_resetData(model,data);data.qpos[:3]=[*xy,.16]
        for i in range(90):
            for j in range(17):mujoco.mj_step(model,data)
            renderer.update_scene(data,camera='source')
            image=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
            cv2.putText(image,label+' | gravity + collisions',(12,28),0,.6,(20,90,230),2)
            writer.write(image)
    writer.release();renderer.close()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(output/'preview-raw.mp4'),
                    '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'preview.mp4')],check=True)


def view(output):
    import mujoco
    import mujoco.viewer
    import time
    model=mujoco.MjModel.from_xml_path(str((output/'scene.xml').resolve()))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer.cam.lookat[:]=[.04,.01,.025];viewer.cam.distance=.7
        viewer.cam.azimuth=120;viewer.cam.elevation=-50
        while viewer.is_running():
            start=time.monotonic();mujoco.mj_step(model,data);viewer.sync()
            time.sleep(max(0.,model.opt.timestep-(time.monotonic()-start)))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['build','verify','record','view'])
    p.add_argument('output',type=Path)
    p.add_argument('--video',type=Path)
    p.add_argument('--render',action='store_true')
    a=p.parse_args()
    if a.stage=='build':build(a.video,a.output)
    elif a.stage=='verify':verify(a.output,a.render)
    elif a.stage=='record':record(a.output)
    else:view(a.output)


if __name__=='__main__':main()
