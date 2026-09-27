#!/usr/bin/env python3
"""Interactive six-region camera calibration for mechArm experiment 3.

This utility only reads a V4L2 camera and writes a YAML region configuration.
It never connects to or commands the robot.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np
import yaml


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>实验三六区域标定</title>
  <style>
    :root { color-scheme: dark; --green:#59e391; --yellow:#ffd166; --red:#ff6b6b; }
    * { box-sizing:border-box; }
    body { margin:0; background:#101214; color:#eef1f4; font-family:"Microsoft YaHei",system-ui,sans-serif; }
    main { width:min(1500px,100%); margin:auto; padding:12px; }
    h1 { margin:0 0 5px; font-size:22px; }
    .notice { margin:0 0 10px; color:#c8d0d8; }
    .layout { display:grid; grid-template-columns:minmax(0,1fr) 290px; gap:12px; align-items:start; }
    .viewport { position:relative; width:100%; background:#000; border:1px solid #3d4650; line-height:0; }
    #video { display:block; width:100%; height:auto; }
    #overlay { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; }
    aside { background:#1a1e22; border:1px solid #394049; border-radius:8px; padding:12px; }
    #status { min-height:48px; color:var(--yellow); line-height:1.45; }
    .regions { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin:8px 0 12px; }
    .region { padding:8px 4px; border:1px solid #59636e; border-radius:6px; background:#252b31; color:#fff; }
    .region.active { outline:2px solid var(--yellow); }
    .region.done { background:#164d32; border-color:var(--green); }
    button { width:100%; padding:9px 8px; margin:4px 0; border:0; border-radius:6px; background:#34404b; color:#fff; font-size:14px; cursor:pointer; }
    button.primary { background:#197a47; }
    button.warn { background:#8a4a16; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .small { color:#aeb7c0; font-size:13px; line-height:1.5; }
    .ok { color:var(--green)!important; }
    .error { color:var(--red)!important; }
    @media(max-width:900px) { .layout { grid-template-columns:1fr; } }
  </style>
</head>
<body><main>
  <h1>实验三六区域标定</h1>
  <p class="notice">只读取摄像头，不连接机械臂。按区域 1→6，每个区域从左上角开始顺时针点击四个角点。</p>
  <div class="layout">
    <div class="viewport">
      <img id="video" src="/stream.mjpg" alt="Jetson 实时摄像头">
      <canvas id="overlay" width="1280" height="720"></canvas>
    </div>
    <aside>
      <div id="status">正在读取摄像头……</div>
      <div class="regions" id="regionButtons"></div>
      <button id="undo">撤销最后一个点</button>
      <button id="redraw" class="warn">重画当前区域</button>
      <button id="reset" class="warn">清空全部区域</button>
      <button id="mode">切换到测试模式</button>
      <button id="save" class="primary" disabled>保存 regions.yaml</button>
      <p class="small">
        标定模式：每区点击四点。<br>
        测试模式：点击任意位置，显示 GRID 1～6 或 OUTSIDE。<br>
        保存前会检查图像边界、区域面积和互相重叠。
      </p>
    </aside>
  </div>
<script>
const W=1280, H=720, COLORS=['#ff5252','#40c4ff','#ffd740','#69f0ae','#e040fb','#ff9100'];
let regions=Array.from({length:6},(_,i)=>({id:i+1,polygon:[]}));
let active=1, mode='calibrate', testPoint=null, loaded=false;
const canvas=document.getElementById('overlay'), ctx=canvas.getContext('2d');
const status=document.getElementById('status'), save=document.getElementById('save');

function message(text,kind='') { status.textContent=text; status.className=kind; }
function activeRegion(){ return regions[active-1]; }
function complete(){ return regions.every(r=>r.polygon.length===4); }
function nextIncomplete(){ const r=regions.find(r=>r.polygon.length<4); return r?r.id:null; }
function pointInPolygon(p,poly){
  let inside=false;
  for(let i=0,j=poly.length-1;i<poly.length;j=i++){
    const xi=poly[i][0],yi=poly[i][1],xj=poly[j][0],yj=poly[j][1];
    const cross=((yi>p[1])!==(yj>p[1])) && (p[0]<(xj-xi)*(p[1]-yi)/(yj-yi)+xi);
    if(cross) inside=!inside;
  }
  return inside;
}
function locate(p){ const matches=regions.filter(r=>r.polygon.length===4&&pointInPolygon(p,r.polygon)); return matches.length===1?matches[0].id:null; }
function renderButtons(){
  const root=document.getElementById('regionButtons'); root.innerHTML='';
  regions.forEach(r=>{
    const b=document.createElement('button'); b.textContent=`区域 ${r.id} (${r.polygon.length}/4)`;
    b.className='region'+(r.id===active?' active':'')+(r.polygon.length===4?' done':'');
    b.onclick=()=>{ active=r.id; render(); message(mode==='calibrate'?`当前选择区域 ${active}`:'测试模式：点击画面检查 grid_id'); };
    root.appendChild(b);
  });
  save.disabled=!complete();
}
function render(){
  ctx.clearRect(0,0,W,H); ctx.lineJoin='round'; ctx.font='bold 24px sans-serif';
  regions.forEach((r,index)=>{
    const pts=r.polygon, color=COLORS[index]; if(!pts.length)return;
    ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=r.id===active?5:3;
    ctx.beginPath(); ctx.moveTo(pts[0][0],pts[0][1]); pts.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));
    if(pts.length===4){ ctx.closePath(); ctx.globalAlpha=.16; ctx.fill(); ctx.globalAlpha=1; }
    ctx.stroke();
    pts.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.fill();ctx.fillText(String(i+1),p[0]+10,p[1]-8);});
    const cx=pts.reduce((s,p)=>s+p[0],0)/pts.length, cy=pts.reduce((s,p)=>s+p[1],0)/pts.length;
    ctx.fillStyle='#fff';ctx.strokeStyle='#000';ctx.lineWidth=5;ctx.strokeText(`GRID ${r.id}`,cx-42,cy);ctx.fillText(`GRID ${r.id}`,cx-42,cy);
  });
  if(testPoint){
    const id=locate(testPoint), label=id?`GRID ${id}`:'OUTSIDE';
    ctx.fillStyle=id?'#59e391':'#ff6b6b';ctx.beginPath();ctx.arc(testPoint[0],testPoint[1],10,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle='#000';ctx.lineWidth=5;ctx.strokeText(label,testPoint[0]+14,testPoint[1]-12);ctx.fillText(label,testPoint[0]+14,testPoint[1]-12);
  }
  renderButtons();
}
canvas.addEventListener('click',event=>{
  const box=canvas.getBoundingClientRect();
  const p=[Math.round((event.clientX-box.left)*W/box.width),Math.round((event.clientY-box.top)*H/box.height)];
  if(mode==='test'){
    testPoint=p; const id=locate(p); render(); message(id?`合法：像素 (${p[0]}, ${p[1]}) 属于 GRID ${id}`:`像素 (${p[0]}, ${p[1]}) 位于六区之外`,id?'ok':'error'); return;
  }
  const r=activeRegion(); if(r.polygon.length>=4){ message(`区域 ${active} 已完成；请选择未完成区域或点击“重画当前区域”`,'error'); return; }
  r.polygon.push(p); testPoint=null;
  if(r.polygon.length===4){ const next=nextIncomplete(); if(next){active=next;message(`区域 ${r.id} 完成，现在标定区域 ${active}`,'ok');}else{message('六个区域均已完成，请先切换测试模式抽查，再保存。','ok');} }
  else message(`区域 ${active}：已记录 ${r.polygon.length}/4 个角点`);
  render();
});
document.getElementById('undo').onclick=()=>{
  if(mode==='test'){testPoint=null;render();message('测试标记已清除');return;}
  let r=activeRegion(); if(!r.polygon.length){ const done=[...regions].reverse().find(x=>x.polygon.length); if(done){active=done.id;r=done;} }
  r.polygon.pop(); render(); message(`区域 ${active}：已记录 ${r.polygon.length}/4 个角点`);
};
document.getElementById('redraw').onclick=()=>{activeRegion().polygon=[];testPoint=null;mode='calibrate';render();message(`区域 ${active} 已清空，请重新点击四个角点`);};
document.getElementById('reset').onclick=()=>{regions.forEach(r=>r.polygon=[]);active=1;testPoint=null;mode='calibrate';render();message('全部区域已清空，从区域 1 开始');};
document.getElementById('mode').onclick=event=>{
  if(mode==='calibrate'&&!complete()){message('必须先完成六个区域才能进入测试模式','error');return;}
  mode=mode==='calibrate'?'test':'calibrate';testPoint=null;
  event.target.textContent=mode==='test'?'返回标定模式':'切换到测试模式';
  render();message(mode==='test'?'测试模式：点击六个区域和区域外位置检查结果':'标定模式：可选择并重画区域');
};
save.onclick=async()=>{
  if(!complete())return;
  save.disabled=true;message('正在校验并保存……');
  try{
    const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_width:W,image_height:H,regions})});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||`HTTP ${response.status}`);
    message(`保存成功：${result.path}`,'ok');mode='test';document.getElementById('mode').textContent='返回标定模式';
  }catch(error){message(`保存失败：${error.message}`,'error');}
  render();
};
async function loadExisting(){
  try{
    const response=await fetch('/api/config',{cache:'no-store'}); const data=await response.json();
    if(response.ok&&Array.isArray(data.regions)&&data.regions.length===6){regions=data.regions;active=1;loaded=true;message('已加载现有 regions.yaml；可测试或重画。','ok');}
    else message('未找到旧配置，从区域 1 开始标定。');
  }catch(_){message('未找到旧配置，从区域 1 开始标定。');}
  render();
}
document.getElementById('video').onload=()=>{if(!loaded)message('画面已连接，从区域 1 左上角开始顺时针点击四个角点。','ok');};
loadExisting();
</script>
</main></body></html>"""


def polygon_area(points: list[list[int]]) -> float:
    array = np.asarray(points, dtype=np.float32)
    return float(abs(cv2.contourArea(array)))


def canonical_polygon(points: Any, width: int, height: int) -> list[list[int]]:
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("每个区域必须恰好包含四个角点")
    parsed: list[list[int]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("角点格式必须为 [x, y]")
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(f"角点 ({x}, {y}) 超出 {width}x{height} 图像边界")
        parsed.append([x, y])
    if len({tuple(point) for point in parsed}) != 4:
        raise ValueError("区域角点不能重复")
    hull = cv2.convexHull(np.asarray(parsed, dtype=np.int32)).reshape(-1, 2)
    if len(hull) != 4:
        raise ValueError("四个角点必须构成有效凸四边形")
    result = [[int(x), int(y)] for x, y in hull]
    if polygon_area(result) < 1000.0:
        raise ValueError("区域面积过小，请框选完整取物范围")
    return result


def validate_regions(payload: dict[str, Any]) -> dict[str, Any]:
    width = int(payload.get("image_width", 0))
    height = int(payload.get("image_height", 0))
    if (width, height) != (1280, 720):
        raise ValueError("标定分辨率必须为 1280x720")
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, list) or len(raw_regions) != 6:
        raise ValueError("必须完整标定六个区域")

    normalized: list[dict[str, Any]] = []
    ids: list[int] = []
    for region in raw_regions:
        if not isinstance(region, dict):
            raise ValueError("区域配置必须是对象")
        region_id = int(region.get("id", 0))
        ids.append(region_id)
        polygon = canonical_polygon(region.get("polygon"), width, height)
        normalized.append({"id": region_id, "polygon": polygon})
    if sorted(ids) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("区域编号必须且只能是 1 到 6")
    normalized.sort(key=lambda item: item["id"])

    for left_index, left in enumerate(normalized):
        left_poly = np.asarray(left["polygon"], dtype=np.float32)
        for right in normalized[left_index + 1 :]:
            right_poly = np.asarray(right["polygon"], dtype=np.float32)
            overlap, _ = cv2.intersectConvexConvex(left_poly, right_poly)
            if overlap > 2.0:
                raise ValueError(f"区域 {left['id']} 与区域 {right['id']} 存在重叠")

    return {
        "format": "mecharm_experiment3_regions/v1",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "camera": {
            "device": "/dev/video0",
            "image_width": width,
            "image_height": height,
            "pixel_format": "MJPG",
        },
        "regions": normalized,
    }


def locate_grid(point: tuple[float, float], regions: list[dict[str, Any]]) -> int | None:
    matches = []
    for region in regions:
        polygon = np.asarray(region["polygon"], dtype=np.float32)
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            matches.append(int(region["id"]))
    return matches[0] if len(matches) == 1 else None


class Camera:
    def __init__(self, device: str, width: int, height: int, fps: int) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.lock = threading.Condition()
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.running = True
        self.error: str | None = None
        self.thread = threading.Thread(target=self._capture, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        with self.lock:
            self.lock.notify_all()
        self.thread.join(timeout=3)

    def _capture(self) -> None:
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.error = f"无法打开摄像头 {self.device}"
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        failures = 0
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 30:
                        self.error = "摄像头连续读取失败"
                        break
                    time.sleep(0.03)
                    continue
                failures = 0
                encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
                if not encoded:
                    continue
                with self.lock:
                    self.jpeg = buffer.tobytes()
                    self.sequence += 1
                    self.lock.notify_all()
        finally:
            cap.release()
            with self.lock:
                self.lock.notify_all()

    def wait_for_frame(self, previous: int) -> tuple[int, bytes | None]:
        with self.lock:
            self.lock.wait_for(
                lambda: self.sequence != previous or not self.running or self.error,
                timeout=2,
            )
            return self.sequence, self.jpeg


class CalibrationServer(ThreadingHTTPServer):
    camera: Camera
    output: Path


class Handler(BaseHTTPRequestHandler):
    server: CalibrationServer

    def send_bytes(self, status: HTTPStatus, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, status: HTTPStatus, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_bytes(status, "application/json; charset=utf-8", payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/calibrate", "/index.html"):
            self.send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            return
        if self.path == "/health":
            if self.server.camera.jpeg:
                self.send_json(HTTPStatus.OK, {"status": "ok", "width": 1280, "height": 720})
            else:
                self.send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"status": "waiting", "error": self.server.camera.error},
                )
            return
        if self.path == "/api/config":
            if not self.server.output.exists():
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "配置尚未保存"})
                return
            try:
                document = yaml.safe_load(self.server.output.read_text(encoding="utf-8"))
                self.send_json(HTTPStatus.OK, document)
            except Exception as error:
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
            return
        if self.path != "/stream.mjpg":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        sequence = -1
        try:
            while self.server.camera.running:
                sequence, frame = self.server.camera.wait_for_frame(sequence)
                if frame is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("请求大小无效")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            document = validate_regions(payload)
            self.server.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.server.output.with_suffix(".yaml.tmp")
            temporary.write_text(
                yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            temporary.replace(self.server.output)
            self.send_json(
                HTTPStatus.OK,
                {"saved": True, "path": str(self.server.output), "regions": document["regions"]},
            )
        except Exception as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"saved": False, "error": str(error)})

    def log_message(self, format: str, *args: object) -> None:
        return


def self_test() -> None:
    regions = []
    for index in range(6):
        column, row = index % 3, index // 3
        x, y = 40 + column * 220, 40 + row * 250
        regions.append(
            {"id": index + 1, "polygon": [[x, y], [x + 160, y], [x + 160, y + 160], [x, y + 160]]}
        )
    document = validate_regions({"image_width": 1280, "image_height": 720, "regions": regions})
    assert len(document["regions"]) == 6
    assert locate_grid((100.0, 100.0), document["regions"]) == 1
    assert locate_grid((1200.0, 700.0), document["regions"]) is None
    overlapping = [dict(region) for region in regions]
    overlapping[1] = {"id": 2, "polygon": [[100, 100], [260, 100], [260, 260], [100, 260]]}
    try:
        validate_regions({"image_width": 1280, "image_height": 720, "regions": overlapping})
    except ValueError as error:
        assert "重叠" in str(error)
    else:
        raise AssertionError("overlapping regions were accepted")
    print("[SELF-TEST] region validation passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 six-region camera calibrator")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/nvidia/mecharm_tasks/experiment3/config/regions.yaml"),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if (args.width, args.height) != (1280, 720):
        raise SystemExit("本实验标定程序固定使用 1280x720")

    camera = Camera(args.device, args.width, args.height, args.fps)
    server = CalibrationServer(("0.0.0.0", args.port), Handler)
    server.camera = camera
    server.output = args.output

    def stop_server(_signum: int, _frame: object) -> None:
        camera.running = False
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    camera.start()
    print(f"[CAMERA] {args.device} 1280x720@{args.fps}", flush=True)
    print(f"[OUTPUT] {args.output}", flush=True)
    print(f"[OPEN] http://0.0.0.0:{args.port}/calibrate", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        camera.stop()
        server.server_close()


if __name__ == "__main__":
    main()
