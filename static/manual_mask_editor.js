/**
 * 手动遮盖编辑器 - 完整前端实现
 * 功能：三层 mask 系统 + SAM2 点击分割 + SAM2 框选（点按式）
 * 特性：鼠标悬停显示十字分割线、点击确定起点/终点
 */

(function() {
    'use strict';

    // ========== 全局状态 ==========
    let mmImages = [];
    let mmFolder = '';
    let mmIdx = -1;
    let mmImg = null;
    let mmLayers = { auto: null, manual: null, inverse: null };
    let mmHistory = [];
    let mmScale = 1.0;
    let mmBrushSize = 15;
    let mmDrawing = false;
    let mmErasing = false;
    let mmLastX = 0, mmLastY = 0;
    let mmTool = 'brush';
    let mmTarget = 'manual';
    let mmLassoPoints = [];
    let mmLassoDrawing = false;
    let mmLassoMousePos = null;
    let mmEKeyDown = false;
    let mmSpaceDown = false;
    let mmOffsetX = 0, mmOffsetY = 0;
    let mmMaskCanvas = null;
    let mmPanning = false;
    let mmPanStartX = 0, mmPanStartY = 0;
    let mmPanStartOffsetX = 0, mmPanStartOffsetY = 0;

    // SAM2 状态
    let mmSam2Points = [];
    let mmSam2Labels = [];
    let mmSam2Loading = false;
    let mmSam2PreviewLayer = null;
    let mmSam2Mode = 'point';
    let mmSam2Positive = true;
    let mmBboxStart = null;
    let mmBboxRect = null;
    let mmBboxDrawing = false;
    let mmBboxState = 'idle';
    let mmBboxStartX = 0, mmBboxStartY = 0;
    let mmBboxEndX = 0, mmBboxEndY = 0;
    let mmMouseX = -1, mmMouseY = -1;
    let mmShowCrosshair = true;  // 十字准线开关

    // 填充容差
    let mmFillTolerance = 30;

    // ========== 初始化 ==========
    function initManualMaskEditor() {
        // 确保 DOM 加载完成
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupManualMaskEvents);
        } else {
            setupManualMaskEvents();
        }
    }

    function setupManualMaskEvents() {
        const canvas = document.getElementById('mm_canvas');
        if (!canvas) return;

        canvas.addEventListener('mousedown', mmMouseDown);
        canvas.addEventListener('mousemove', mmMouseMove);
        canvas.addEventListener('mouseup', mmMouseUp);
        canvas.addEventListener('mouseleave', mmMouseLeave);
        canvas.addEventListener('wheel', mmWheel, { passive: false });
        canvas.addEventListener('contextmenu', e => e.preventDefault());

        // 全局事件
        document.addEventListener('keydown', mmKeyDown);
        document.addEventListener('keyup', mmKeyUp);
        document.addEventListener('mouseup', mmGlobalMouseUp);
    }

    // ========== 辅助函数 ==========
    function mmGetActiveLayer() {
        return mmLayers[mmTarget];
    }

    function mmToImgCoords(ex, ey) {
        const canvas = document.getElementById('mm_canvas');
        if (!canvas) return { x: 0, y: 0 };
        const rect = canvas.getBoundingClientRect();
        const x = (ex - rect.left) / mmScale;
        const y = (ey - rect.top) / mmScale;
        return { x, y };
    }

    function mmUpdateTransform() {
        const canvas = document.getElementById('mm_canvas');
        if (!canvas) return;
        canvas.style.transform = `translate(${mmOffsetX}px, ${mmOffsetY}px)`;
    }

    function mmSetBrush(val) {
        mmBrushSize = Math.max(3, Math.min(100, parseInt(val, 10) || 15));
        const slider = document.getElementById('mm_brush');
        const label = document.getElementById('mm_brush_val');
        if (slider) slider.value = mmBrushSize;
        if (label) label.textContent = mmBrushSize;
    }

    function mmSetFillTolerance(val) {
        mmFillTolerance = Math.max(0, Math.min(100, parseInt(val, 10) || 30));
        const slider = document.getElementById('mm_fill_tol');
        const label = document.getElementById('mm_fill_tol_val');
        if (slider) slider.value = mmFillTolerance;
        if (label) label.textContent = mmFillTolerance;
    }

    function mmSetZoom(val) {
        const container = document.getElementById('mm_canvas_container');
        const canvas = document.getElementById('mm_canvas');
        if (!container || !canvas || !mmImg) return;
        
        const oldScale = mmScale;
        mmScale = Math.max(0.2, Math.min(3.0, parseInt(val, 10) / 100));
        
        const centerX = container.clientWidth / 2;
        const centerY = container.clientHeight / 2;
        const imgX = (centerX - mmOffsetX) / oldScale;
        const imgY = (centerY - mmOffsetY) / oldScale;
        
        canvas.width = mmImg.naturalWidth * mmScale;
        canvas.height = mmImg.naturalHeight * mmScale;
        mmOffsetX = centerX - imgX * mmScale;
        mmOffsetY = centerY - imgY * mmScale;
        
        const slider = document.getElementById('mm_zoom');
        const label = document.getElementById('mm_zoom_val');
        if (slider) slider.value = parseInt(mmScale * 100);
        if (label) label.textContent = parseInt(mmScale * 100);
        
        mmUpdateTransform();
        mmRender();
    }

    function mmZoomAt(screenX, screenY, factor) {
        const canvas = document.getElementById('mm_canvas');
        if (!canvas || !mmImg) return;
        
        const oldScale = mmScale;
        mmScale = Math.max(0.2, Math.min(3.0, mmScale * factor));
        
        const imgX = (screenX - mmOffsetX) / oldScale;
        const imgY = (screenY - mmOffsetY) / oldScale;
        
        canvas.width = mmImg.naturalWidth * mmScale;
        canvas.height = mmImg.naturalHeight * mmScale;
        mmOffsetX = screenX - imgX * mmScale;
        mmOffsetY = screenY - imgY * mmScale;
        
        const slider = document.getElementById('mm_zoom');
        const label = document.getElementById('mm_zoom_val');
        if (slider) slider.value = parseInt(mmScale * 100);
        if (label) label.textContent = parseInt(mmScale * 100);
        
        mmUpdateTransform();
        mmRender();
    }

    // ========== 工具切换 ==========
    function mmSetTool(tool) {
        mmTool = tool;
        mmLassoDrawing = false;
        mmLassoPoints = [];
        mmLassoMousePos = null;
        
        const btns = {
            brush: 'mm_tool_brush',
            lasso: 'mm_tool_lasso',
            fill: 'mm_tool_fill',
            eraser: 'mm_tool_eraser',
            sam2: 'mm_tool_sam2'
        };
        const colors = {
            brush: '#2563eb',
            lasso: '#7c3aed',
            fill: '#16a34a',
            eraser: '#ea580c',
            sam2: '#cc6600'
        };
        
        for (const [t, id] of Object.entries(btns)) {
            const btn = document.getElementById(id);
            if (!btn) continue;
            const active = t === mmTool;
            btn.style.borderColor = active ? colors[t] : '#d0d4da';
            btn.style.background = active ? colors[t] : '#fff';
            btn.style.color = active ? '#fff' : '#555';
        }
        
        const sam2Panel = document.getElementById('mm_sam2_panel');
        if (sam2Panel) sam2Panel.style.display = tool === 'sam2' ? 'block' : 'none';
        
        const sam2LayerLabel = document.getElementById('mm_layer_sam2_label');
        if (sam2LayerLabel) {
            sam2LayerLabel.style.display = tool === 'sam2' ? 'flex' : 'none';
        }
        
        mmRender();
    }

    function mmSetTarget(target) {
        mmTarget = target;
        const manualBtn = document.getElementById('mm_target_manual');
        const inverseBtn = document.getElementById('mm_target_inverse');
        if (manualBtn) {
            manualBtn.style.borderColor = target === 'manual' ? '#eab308' : '#d0d4da';
            manualBtn.style.background = target === 'manual' ? '#fef9c3' : '#fff';
            manualBtn.style.color = target === 'manual' ? '#854d0e' : '#555';
        }
        if (inverseBtn) {
            inverseBtn.style.borderColor = target === 'inverse' ? '#22c55e' : '#d0d4da';
            inverseBtn.style.background = target === 'inverse' ? '#dcfce7' : '#fff';
            inverseBtn.style.color = target === 'inverse' ? '#15803d' : '#555';
        }
    }

    // ========== 文件夹加载 ==========
    function mmLoadFolder() {
        const folder = document.getElementById('mm_folder').value.trim();
        if (!folder) { alert('请输入文件夹路径'); return; }
        
        const fd = new FormData();
        fd.append('folder', folder);
        const outputFolder = document.getElementById('mm_output_folder');
        const autoMaskDir = document.getElementById('mm_auto_mask_dir');
        if (outputFolder && outputFolder.value) fd.append('output_folder', outputFolder.value);
        if (autoMaskDir && autoMaskDir.value) fd.append('auto_mask_dir', autoMaskDir.value);
        
        fetch('/manual_mask/list', { method: 'POST', body: fd })
            .then(r => r.json())
            .then(data => {
                if (data.error) { alert(data.error); return; }
                mmImages = data.images || [];
                mmFolder = data.folder || folder;
                const hasMask = data.has_mask || {};
                document.getElementById('mm_status').textContent = mmImages.length + ' 张图片';
                
                const thumbContainer = document.getElementById('mm_thumbnails');
                if (mmImages.length === 0) {
                    thumbContainer.innerHTML = '<span style="color:#bbb;">文件夹中没有图片</span>';
                    return;
                }
                
                let html = '';
                mmImages.forEach((fname, i) => {
                    const imgPath = mmFolder + '\\' + fname;
                    const imgUrl = '/manual_mask/image?path=' + encodeURIComponent(imgPath);
                    const masked = hasMask[fname];
                    html += '<div onclick="mmSelect(' + i + ')" id="mm_thumb_' + i + '" style="flex-shrink:0;width:60px;height:60px;border:2px solid #ddd;border-radius:4px;overflow:hidden;cursor:pointer;position:relative;">';
                    html += '<img src="' + imgUrl + '" style="width:100%;height:100%;object-fit:cover;">';
                    if (masked) {
                        html += '<span style="position:absolute;top:2px;right:2px;background:#16a34a;color:#fff;font-size:10px;padding:1px 4px;border-radius:2px;">✓</span>';
                    }
                    html += '</div>';
                });
                thumbContainer.innerHTML = html;
                if (mmImages.length > 0) mmSelect(0);
            })
            .catch(err => alert('加载失败: ' + err));
    }

    function mmSelect(idx) {
        if (idx < 0 || idx >= mmImages.length) return;
        mmIdx = idx;
        
        document.querySelectorAll('#mm_thumbnails > div').forEach((el, i) => {
            el.style.borderColor = (i === idx) ? '#2563eb' : '#ddd';
        });
        
        const fname = mmImages[idx];
        const imgPath = mmFolder + '\\' + fname;
        mmImg = new Image();
        mmImg.crossOrigin = 'anonymous';
        mmImg.onload = function() {
            document.getElementById('mm_no_image').style.display = 'none';
            mmSetupCanvas();
            mmLoadMasks(imgPath);
        };
        mmImg.src = '/manual_mask/image?path=' + encodeURIComponent(imgPath);
        document.getElementById('mm_status').textContent = (idx + 1) + '/' + mmImages.length + ' ' + fname;
    }

    function mmSetupCanvas() {
        const canvas = document.getElementById('mm_canvas');
        const container = document.getElementById('mm_canvas_container');
        if (!mmImg || !canvas || !container) return;
        
        const iw = mmImg.naturalWidth, ih = mmImg.naturalHeight;
        
        ['auto', 'manual', 'inverse'].forEach(layer => {
            mmLayers[layer] = document.createElement('canvas');
            mmLayers[layer].width = iw;
            mmLayers[layer].height = ih;
            const ctx = mmLayers[layer].getContext('2d');
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, iw, ih);
        });
        mmMaskCanvas = mmLayers.manual;
        mmHistory = [];
        
        const maxW = container.clientWidth - 20;
        const maxH = container.clientHeight - 20;
        mmScale = Math.min(1, maxW / iw, maxH / ih);
        canvas.width = iw * mmScale;
        canvas.height = ih * mmScale;
        mmOffsetX = (container.clientWidth - canvas.width) / 2;
        mmOffsetY = (container.clientHeight - canvas.height) / 2;
        
        mmUpdateTransform();
        mmRender();
    }

    function mmLoadMasks(imgPath) {
        const outputFolder = document.getElementById('mm_output_folder');
        const autoMaskDir = document.getElementById('mm_auto_mask_dir');
        
        let loaded = 0;
        const totalLayers = 3;
        const onLayerDone = () => {
            loaded++;
            if (loaded >= totalLayers) {
                mmPushHistory();
                mmRender();
            }
        };
        
        ['auto', 'manual', 'inverse'].forEach(layer => {
            let url = '/manual_mask/load?path=' + encodeURIComponent(imgPath) + '&layer=' + layer;
            if (outputFolder && outputFolder.value) url += '&output_folder=' + encodeURIComponent(outputFolder.value);
            if (autoMaskDir && autoMaskDir.value) url += '&auto_mask_dir=' + encodeURIComponent(autoMaskDir.value);
            
            fetch(url)
                .then(r => r.ok ? r.blob() : null)
                .then(blob => {
                    if (blob && blob.size > 100) {
                        const blobUrl = URL.createObjectURL(blob);
                        const maskImg = new Image();
                        maskImg.onload = () => {
                            const ctx = mmLayers[layer].getContext('2d');
                            ctx.clearRect(0, 0, mmLayers[layer].width, mmLayers[layer].height);
                            ctx.drawImage(maskImg, 0, 0, mmLayers[layer].width, mmLayers[layer].height);
                            URL.revokeObjectURL(blobUrl);
                            onLayerDone();
                        };
                        maskImg.onerror = onLayerDone;
                        maskImg.src = blobUrl;
                    } else {
                        onLayerDone();
                    }
                })
                .catch(onLayerDone);
        });
    }

    // ========== 鼠标事件 ==========
    function mmMouseDown(e) {
        if (!mmLayers.manual) return;
        e.preventDefault();
        
        if (mmSpaceDown && e.button === 0) {
            mmPanning = true;
            mmPanStartX = e.clientX;
            mmPanStartY = e.clientY;
            mmPanStartOffsetX = mmOffsetX;
            mmPanStartOffsetY = mmOffsetY;
            document.getElementById('mm_canvas').style.cursor = 'grabbing';
            return;
        }
        
        const pos = mmToImgCoords(e.clientX, e.clientY);
        
        // 填充工具
        if (mmTool === 'fill') {
            if (e.button === 0) mmFloodFill(pos.x, pos.y);
            return;
        }
        
        // SAM2 工具
        if (mmTool === 'sam2') {
            if (e.button !== 0) return;
            
            if (mmSam2Mode === 'bbox') {
                // 点按式框选：左键确定起点
                if (e.shiftKey) {
                    // Shift+左键切换正/反向
                    mmSam2Positive = !mmSam2Positive;
                    mmSam2UpdatePanel();
                    return;
                }
                
                if (mmBboxState === 'idle') {
                    mmBboxStartX = pos.x;
                    mmBboxStartY = pos.y;
                    mmBboxEndX = pos.x;
                    mmBboxEndY = pos.y;
                    mmBboxState = 'waiting_end';
                } else if (mmBboxState === 'waiting_end') {
                    mmBboxEndX = pos.x;
                    mmBboxEndY = pos.y;
                    mmFinalizeBbox();
                    // 完成后继续绘制下一个框
                    mmBboxStartX = pos.x;
                    mmBboxStartY = pos.y;
                    mmBboxEndX = pos.x;
                    mmBboxEndY = pos.y;
                    mmBboxState = 'waiting_end';
                }
                mmSam2UpdatePanel();
                mmRender();
                return;
            } else {
                // 点选模式
                const label = mmSam2Positive ? 1 : 0;
                mmSam2Points.push([Math.round(pos.x), Math.round(pos.y)]);
                mmSam2Labels.push(label);
                mmSam2UpdatePanel();
                mmRender();
                mmSam2Predict();
                return;
            }
        }
        
        // 套索工具
        if (mmTool === 'lasso') {
            if (e.button === 2) {
                if (mmLassoDrawing && mmLassoPoints.length >= 3) {
                    mmLassoDrawing = false;
                    mmLassoMousePos = null;
                    mmFillLasso();
                }
                return;
            }
            if (e.button !== 0) return;
            
            if (!mmLassoDrawing) {
                mmPushHistory();
                mmLassoDrawing = true;
                mmLassoPoints = [pos];
            } else {
                mmLassoPoints.push(pos);
            }
            mmRender();
            return;
        }
        
        // 画笔/橡皮
        const isEraser = mmTool === 'eraser' || mmEKeyDown;
        mmDrawing = (e.button === 0 && !isEraser);
        mmErasing = (e.button === 2 || (e.button === 0 && isEraser));
        if (!mmDrawing && !mmErasing) return;
        mmLastX = pos.x;
        mmLastY = pos.y;
        mmPushHistory();
        mmDrawLine(pos.x, pos.y, pos.x, pos.y);
    }

    function mmMouseMove(e) {
        if (!mmLayers.manual) return;
        
        if (mmPanning) {
            mmOffsetX = mmPanStartOffsetX + (e.clientX - mmPanStartX);
            mmOffsetY = mmPanStartOffsetY + (e.clientY - mmPanStartY);
            mmUpdateTransform();
            return;
        }
        
        const pos = mmToImgCoords(e.clientX, e.clientY);
        
        // 更新十字准线位置
        mmMouseX = pos.x;
        mmMouseY = pos.y;
        mmShowCrosshair = true;
        
        // SAM2 框选模式：实时更新预览框
        if (mmTool === 'sam2' && mmSam2Mode === 'bbox' && mmBboxState === 'waiting_end') {
            mmBboxEndX = pos.x;
            mmBboxEndY = pos.y;
            mmRender();
            return;
        }
        
        // 套索：跟随线
        if (mmTool === 'lasso') {
            mmLassoMousePos = pos;
            if (mmLassoDrawing) mmRender();
            return;
        }
        
        // 画笔/橡皮
        if (!mmDrawing && !mmErasing) {
            mmRender();
            return;
        }
        mmDrawLine(mmLastX, mmLastY, pos.x, pos.y);
        mmLastX = pos.x;
        mmLastY = pos.y;
    }

    function mmMouseUp(e) {
        if (mmPanning) {
            mmPanning = false;
            document.getElementById('mm_canvas').style.cursor = mmSpaceDown ? 'grab' : 'crosshair';
            return;
        }
        
        if (mmTool === 'lasso') return;
        
        mmDrawing = false;
        mmErasing = false;
    }

    function mmMouseLeave(e) {
        mmPanning = false;
        mmDrawing = false;
        mmErasing = false;
        mmShowCrosshair = false;
        mmRender();
    }

    function mmGlobalMouseUp() {
        mmDrawing = false;
        mmErasing = false;
        if (mmPanning) {
            mmPanning = false;
            const canvas = document.getElementById('mm_canvas');
            if (canvas) canvas.style.cursor = mmSpaceDown ? 'grab' : 'crosshair';
        }
    }

    function mmWheel(e) {
        e.preventDefault();
        if (e.shiftKey) {
            const delta = e.deltaY < 0 ? 2 : -2;
            mmSetBrush(mmBrushSize + delta);
        } else {
            const rect = document.getElementById('mm_canvas').getBoundingClientRect();
            mmZoomAt(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.1 : 0.9);
        }
    }

    // ========== 键盘事件 ==========
    function mmKeyDown(e) {
        const tab = document.getElementById('manualmask-tab');
        if (!tab || !tab.classList.contains('active')) return;
        
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'z') { e.preventDefault(); mmUndo(); }
            if (e.key === 's') { e.preventDefault(); mmMergeSave(false); }
            if (e.key === 'x') { e.preventDefault(); mmMergeSave(true); }
            return;
        }
        
        // 工具快捷键
        if (e.key === 'b' || e.key === 'B') { e.preventDefault(); mmSetTool('brush'); }
        if (e.key === 'l' || e.key === 'L') { e.preventDefault(); mmSetTool('lasso'); }
        if (e.key === 'f' || e.key === 'F') { e.preventDefault(); mmSetTool(mmTool === 'brush' ? 'lasso' : 'brush'); }
        if (e.key === 'g' || e.key === 'G') { e.preventDefault(); mmSetTool('fill'); }
        if (e.key === 'w' || e.key === 'W') { e.preventDefault(); mmSetTool('eraser'); }
        
        // SAM2 快捷键
        if (e.key === 'q' || e.key === 'Q') { e.preventDefault(); mmSetTool('sam2'); mmSam2SetMode('point'); mmSam2SetPositive(true); }
        if (e.key === 'e' || e.key === 'E') { e.preventDefault(); mmSetTool('sam2'); mmSam2SetMode('point'); mmSam2SetPositive(false); }
        if (e.key === 'r' || e.key === 'R') { e.preventDefault(); mmSetTool('sam2'); mmSam2SetMode('bbox'); }
        
        // 图层快捷键
        if (e.key === '1') { e.preventDefault(); mmSetTarget('manual'); }
        if (e.key === '2') { e.preventDefault(); mmSetTarget('inverse'); }
        
        // 翻页快捷键
        if (e.key === 'a' || e.key === 'A') { e.preventDefault(); if (!mmDrawing && !mmSam2Loading) mmPrev(); }
        if (e.key === 'd' || e.key === 'D') { e.preventDefault(); if (!mmDrawing && !mmSam2Loading) mmNext(); }
        
        // Space 平移
        if (e.key === ' ' && !mmSpaceDown) {
            e.preventDefault();
            mmSpaceDown = true;
            const canvas = document.getElementById('mm_canvas');
            if (canvas) canvas.style.cursor = 'grab';
        }
        
        // 套索快捷键
        if (e.key === 'Enter' && mmTool === 'lasso' && mmLassoDrawing && mmLassoPoints.length >= 3) {
            e.preventDefault();
            mmLassoDrawing = false;
            mmLassoMousePos = null;
            mmFillLasso();
        }
        if (e.key === 'Escape' && mmTool === 'lasso' && mmLassoDrawing) {
            e.preventDefault();
            mmLassoDrawing = false;
            mmLassoPoints = [];
            mmLassoMousePos = null;
            mmRender();
        }
        
        // ESC 取消 SAM2 框选
        if (e.key === 'Escape' && mmTool === 'sam2' && mmSam2Mode === 'bbox' && mmBboxState === 'waiting_end') {
            e.preventDefault();
            mmBboxState = 'idle';
            mmBboxRect = null;
            mmSam2UpdatePanel();
            mmRender();
        }
    }

    function mmKeyUp(e) {
        if (e.key === 'e' || e.key === 'E') { mmEKeyDown = false; }
        if (e.key === ' ') {
            mmSpaceDown = false;
            mmPanning = false;
            const canvas = document.getElementById('mm_canvas');
            if (canvas) canvas.style.cursor = 'crosshair';
        }
    }

    // ========== 绘制逻辑 ==========
    function mmDrawLine(x1, y1, x2, y2) {
        const activeLayer = mmGetActiveLayer();
        if (!activeLayer) return;
        const ctx = activeLayer.getContext('2d');
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = mmBrushSize * 2 + 1;
        ctx.strokeStyle = mmErasing ? 'black' : 'white';
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        mmRender();
    }

    function mmFillLasso() {
        const activeLayer = mmGetActiveLayer();
        if (!activeLayer || mmLassoPoints.length < 3) {
            mmLassoPoints = [];
            mmRender();
            return;
        }
        const ctx = activeLayer.getContext('2d');
        ctx.save();
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.moveTo(mmLassoPoints[0].x, mmLassoPoints[0].y);
        for (let i = 1; i < mmLassoPoints.length; i++) {
            ctx.lineTo(mmLassoPoints[i].x, mmLassoPoints[i].y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.restore();
        mmLassoPoints = [];
        mmRender();
    }

    // 油漆桶填充
    function mmFloodFill(x, y) {
        const activeLayer = mmGetActiveLayer();
        if (!activeLayer || !mmImg) return;
        const w = activeLayer.width, h = activeLayer.height;
        const px = Math.floor(x), py = Math.floor(y);
        if (px < 0 || px >= w || py < 0 || py >= h) return;
        
        // 获取原图像素
        if (!window._mmSrcCanvas) window._mmSrcCanvas = document.createElement('canvas');
        const srcC = window._mmSrcCanvas;
        srcC.width = w; srcC.height = h;
        const srcCtx = srcC.getContext('2d');
        srcCtx.drawImage(mmImg, 0, 0, w, h);
        const srcData = srcCtx.getImageData(0, 0, w, h).data;
        
        const seedIdx = (py * w + px) * 4;
        const sR = srcData[seedIdx], sG = srcData[seedIdx + 1], sB = srcData[seedIdx + 2];
        
        mmPushHistory();
        const ctx = activeLayer.getContext('2d');
        const maskData = ctx.getImageData(0, 0, w, h);
        const md = maskData.data;
        
        const visited = new Uint8Array(w * h);
        const stack = [px, py];
        const tol = mmFillTolerance;
        
        while (stack.length > 0) {
            const cy = stack.pop(), cx = stack.pop();
            if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;
            const ci = cy * w + cx;
            if (visited[ci]) continue;
            visited[ci] = 1;
            
            const si = ci * 4;
            if (Math.abs(srcData[si] - sR) > tol || Math.abs(srcData[si + 1] - sG) > tol || Math.abs(srcData[si + 2] - sB) > tol) continue;
            
            md[ci * 4] = 255;
            md[ci * 4 + 1] = 255;
            md[ci * 4 + 2] = 255;
            md[ci * 4 + 3] = 255;
            
            stack.push(cx + 1, cy, cx - 1, cy, cx, cy + 1, cx, cy - 1);
        }
        ctx.putImageData(maskData, 0, 0);
        mmRender();
    }

    // ========== SAM2 功能 ==========
    function mmSam2SetMode(mode) {
        mmSam2Mode = mode;
        const pointBtn = document.getElementById('mm_sam2_mode_point');
        const bboxBtn = document.getElementById('mm_sam2_mode_bbox');
        if (pointBtn) {
            pointBtn.style.borderColor = mode === 'point' ? '#2563eb' : '#d0d4da';
            pointBtn.style.background = mode === 'point' ? '#2563eb' : '#fff';
            pointBtn.style.color = mode === 'point' ? '#fff' : '#555';
        }
        if (bboxBtn) {
            bboxBtn.style.borderColor = mode === 'bbox' ? '#2563eb' : '#d0d4da';
            bboxBtn.style.background = mode === 'bbox' ? '#2563eb' : '#fff';
            bboxBtn.style.color = mode === 'bbox' ? '#fff' : '#555';
        }
        mmSam2Points = [];
        mmSam2Labels = [];
        mmBboxStart = null;
        mmBboxRect = null;
        mmBboxDrawing = false;
        mmBboxState = 'idle';
        mmSam2UpdatePanel();
        mmRender();
    }

    function mmSam2SetPositive(isPositive) {
        mmSam2Positive = isPositive;
        const posBtn = document.getElementById('mm_sam2_mode_positive');
        const negBtn = document.getElementById('mm_sam2_mode_negative');
        if (posBtn) {
            posBtn.style.borderColor = isPositive ? '#16a34a' : '#d0d4da';
            posBtn.style.background = isPositive ? '#dcfce7' : '#fff';
            posBtn.style.color = isPositive ? '#15803d' : '#555';
        }
        if (negBtn) {
            negBtn.style.borderColor = !isPositive ? '#ef4444' : '#d0d4da';
            negBtn.style.background = !isPositive ? '#fef2f2' : '#fff';
            negBtn.style.color = !isPositive ? '#dc2626' : '#555';
        }
        mmSam2UpdatePanel();
    }

    function mmSam2UpdatePanel() {
        const el = document.getElementById('mm_sam2_points');
        if (!el) return;
        
        let html = '';
        
        if (mmSam2Mode === 'bbox' && mmBboxState === 'waiting_end') {
            const x1 = Math.round(Math.min(mmBboxStartX, mmBboxEndX));
            const y1 = Math.round(Math.min(mmBboxStartY, mmBboxEndY));
            const x2 = Math.round(Math.max(mmBboxStartX, mmBboxEndX));
            const y2 = Math.round(Math.max(mmBboxStartY, mmBboxEndY));
            const color = mmSam2Positive ? '#44ff44' : '#ff4444';
            const modeText = mmSam2Positive ? '正向' : '反向';
            html = '<span style="color:#ffaa44;">■ 绘制中...</span> ' + modeText + '<br>';
            html += '<span style="font-size:10px;">起点 [' + Math.round(mmBboxStartX) + ',' + Math.round(mmBboxStartY) + ']</span><br>';
            html += '<span style="font-size:10px;color:' + color + ';">预览 [' + x1 + ',' + y1 + ']-[' + x2 + ',' + y2 + ']</span>';
        } else if (mmSam2Mode === 'bbox') {
            html = '点击确定矩形起点，再点击确定终点<br>';
            html += '<span style="color:#888;">Shift+左键切换正/反向</span>';
        } else if (mmSam2Points.length === 0) {
            html = '点击画布添加点...';
        } else {
            for (let i = 0; i < mmSam2Points.length; i++) {
                const color = mmSam2Labels[i] === 1 ? '#44ff44' : '#ff4444';
                const text = mmSam2Labels[i] === 1 ? '前景' : '背景';
                html += '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + color + ';"></span>';
                html += ' [' + mmSam2Points[i][0] + ',' + mmSam2Points[i][1] + '] ' + text + '<br>';
            }
        }
        el.innerHTML = html;
    }

    function mmFinalizeBbox() {
        const x1 = Math.round(Math.min(mmBboxStartX, mmBboxEndX));
        const y1 = Math.round(Math.min(mmBboxStartY, mmBboxEndY));
        const x2 = Math.round(Math.max(mmBboxStartX, mmBboxEndX));
        const y2 = Math.round(Math.max(mmBboxStartY, mmBboxEndY));
        
        if (x2 - x1 < 5 || y2 - y1 < 5) {
            mmBboxRect = null;
            return;
        }
        
        // 将矩形转换为两个点（SAM2 bbox API）
        mmSam2Points.push([x1, y1]);
        mmSam2Labels.push(mmSam2Positive ? 1 : 0);
        mmSam2Points.push([x2, y2]);
        mmSam2Labels.push(mmSam2Positive ? 1 : 0);
        
        mmBboxRect = [x1, y1, x2, y2];
        mmSam2Predict();
    }

    async function mmSam2Predict() {
        if (mmSam2Points.length === 0) return;
        mmSam2Loading = true;
        
        const status = document.getElementById('mm_sam2_points');
        if (status) status.innerHTML = '<span style="color:#ffaa44;">🔄 SAM2 推理中...</span>';
        
        const modelSelect = document.getElementById('mm_sam2_model');
        const modelName = modelSelect ? modelSelect.value : null;
        
        try {
            let res, data;
            
            if (mmSam2Mode === 'bbox' && mmBboxRect) {
                const [x1, y1, x2, y2] = mmBboxRect;
                res = await fetch('/editor/sam2_bbox', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_path: mmFolder + '\\' + mmImages[mmIdx],
                        bbox: [x1, y1, x2, y2],
                        model_name: modelName
                    })
                });
                data = await res.json();
            } else {
                res = await fetch('/editor/sam2_point', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_path: mmFolder + '\\' + mmImages[mmIdx],
                        points: JSON.stringify(mmSam2Points),
                        labels: JSON.stringify(mmSam2Labels),
                        model_name: modelName
                    })
                });
                data = await res.json();
            }
            
            if (!data.success) {
                if (status) status.innerHTML = '<span style="color:#ff6666;">❌ 失败: ' + (data.error || '未知错误') + '</span>';
                return;
            }
            
            const img = new Image();
            img.onload = () => {
                if (!window._mmSam2Canvas) window._mmSam2Canvas = document.createElement('canvas');
                const c = window._mmSam2Canvas;
                c.width = mmImg.naturalWidth;
                c.height = mmImg.naturalHeight;
                const x = c.getContext('2d');
                x.drawImage(img, 0, 0, c.width, c.height);
                mmSam2PreviewLayer = c;
                
                const sam2LayerItem = document.getElementById('sam2-preview-layer-item');
                if (sam2LayerItem) sam2LayerItem.style.display = 'block';
                
                if (status) status.innerHTML = '<span style="color:#44ff44;">✅ 完成</span>';
                mmSam2UpdatePanel();
                mmRender();
                
                // 框选模式：推理完成后重置状态，允许绘制下一个框
                if (mmSam2Mode === 'bbox') {
                    mmSam2Points = [];
                    mmSam2Labels = [];
                    mmBboxRect = null;
                    mmBboxState = 'idle';
                }
            };
            img.src = data.mask_base64;
        } catch (e) {
            if (status) status.innerHTML = '<span style="color:#ff6666;">❌ 错误: ' + e.message + '</span>';
        } finally {
            mmSam2Loading = false;
        }
    }

    function mmSam2Confirm() {
        if (!mmSam2PreviewLayer || !mmLayers.manual) return;
        mmPushHistory();
        
        const srcCtx = mmSam2PreviewLayer.getContext('2d');
        const srcData = srcCtx.getImageData(0, 0, mmSam2PreviewLayer.width, mmSam2PreviewLayer.height).data;
        const targetCtx = mmLayers.manual.getContext('2d');
        const targetData = targetCtx.getImageData(0, 0, mmLayers.manual.width, mmLayers.manual.height);
        const td = targetData.data;
        
        for (let i = 0; i < td.length; i += 4) {
            if (srcData[i] > 127) {
                td[i] = td[i + 1] = td[i + 2] = 255;
                td[i + 3] = 255;
            }
        }
        targetCtx.putImageData(targetData, 0, 0);
        mmSam2Clear();
        mmRender();
    }

    function mmSam2Clear() {
        mmSam2Points = [];
        mmSam2Labels = [];
        mmBboxRect = null;
        mmBboxState = 'idle';
        mmSam2PreviewLayer = null;
        
        const sam2LayerItem = document.getElementById('sam2-preview-layer-item');
        if (sam2LayerItem) sam2LayerItem.style.display = 'none';
        
        mmSam2UpdatePanel();
        mmRender();
    }

    // ========== 历史记录 ==========
    function mmPushHistory() {
        const activeLayer = mmGetActiveLayer();
        if (!activeLayer) return;
        const ctx = activeLayer.getContext('2d');
        mmHistory.push({
            target: mmTarget,
            data: ctx.getImageData(0, 0, activeLayer.width, activeLayer.height)
        });
        if (mmHistory.length > 30) mmHistory.shift();
    }

    function mmUndo() {
        if (mmHistory.length <= 1) return;
        const entry = mmHistory.pop();
        if (entry && mmLayers[entry.target]) {
            const ctx = mmLayers[entry.target].getContext('2d');
            ctx.putImageData(entry.data, 0, 0);
        }
        mmRender();
    }

    function mmReset() {
        const activeLayer = mmGetActiveLayer();
        if (!activeLayer) return;
        mmPushHistory();
        const ctx = activeLayer.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, activeLayer.width, activeLayer.height);
        mmRender();
    }

    // ========== 保存/合并 ==========
    function mmSaveLayers() {
        if (!mmLayers.manual || mmIdx < 0) return;
        const fname = mmImages[mmIdx];
        const imgPath = mmFolder + '\\' + fname;
        const outputFolder = document.getElementById('mm_output_folder');
        let saved = 0;
        
        ['manual', 'inverse'].forEach(layer => {
            const dataUrl = mmLayers[layer].toDataURL('image/png');
            const fd = new FormData();
            fd.append('image_path', imgPath);
            fd.append('mask_base64', dataUrl);
            fd.append('layer_type', layer);
            if (outputFolder && outputFolder.value) fd.append('output_folder', outputFolder.value);
            
            fetch('/manual_mask/save', { method: 'POST', body: fd })
                .then(r => r.json())
                .then(data => {
                    saved++;
                    if (saved >= 2) {
                        alert('图层已保存');
                        const thumb = document.getElementById('mm_thumb_' + mmIdx);
                        if (thumb && !thumb.querySelector('span')) {
                            thumb.innerHTML += '<span style="position:absolute;top:2px;right:2px;background:#16a34a;color:#fff;font-size:10px;padding:1px 4px;border-radius:2px;">✓</span>';
                        }
                    }
                })
                .catch(err => { saved++; if (saved >= 2) alert('保存失败: ' + err); });
        });
    }

    function mmMergeSave(invert) {
        if (mmIdx < 0 || !mmLayers.manual) return;
        const fname = mmImages[mmIdx];
        const imgPath = mmFolder + '\\' + fname;
        const outputFolder = document.getElementById('mm_output_folder');
        const autoMaskDir = document.getElementById('mm_auto_mask_dir');
        
        const manualB64 = mmLayers.manual.toDataURL('image/png');
        const inverseB64 = mmLayers.inverse.toDataURL('image/png');
        
        const fd = new FormData();
        fd.append('image_path', imgPath);
        fd.append('manual_base64', manualB64);
        fd.append('inverse_base64', inverseB64);
        if (outputFolder && outputFolder.value) fd.append('output_folder', outputFolder.value);
        if (autoMaskDir && autoMaskDir.value) fd.append('auto_mask_dir', autoMaskDir.value);
        if (invert) fd.append('invert', '1');
        
        fetch('/manual_mask/merge', { method: 'POST', body: fd })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    const ratio = (data.mask_ratio * 100).toFixed(1);
                    const mode = data.inverted ? '反向' : '正常';
                    alert(mode + '合并保存完成！遮盖比例: ' + ratio + '%\n保存到: ' + data.final_path);
                } else {
                    alert('合并保存失败: ' + (data.error || ''));
                }
            })
            .catch(err => alert('合并保存请求失败: ' + err));
    }

    function mmMergePreview() {
        if (mmIdx < 0) return;
        mmSaveLayers();
        setTimeout(() => {
            const fname = mmImages[mmIdx];
            const imgPath = mmFolder + '\\' + fname;
            const outputFolder = document.getElementById('mm_output_folder');
            const fd = new FormData();
            fd.append('image_path', imgPath);
            if (outputFolder && outputFolder.value) fd.append('output_folder', outputFolder.value);
            
            fetch('/manual_mask/merge', { method: 'POST', body: fd })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const ratio = (data.mask_ratio * 100).toFixed(1);
                        alert('合并完成！遮盖比例: ' + ratio + '%\n保存到: ' + data.final_path);
                    } else {
                        alert('合并失败: ' + (data.error || ''));
                    }
                })
                .catch(err => alert('合并请求失败: ' + err));
        }, 500);
    }

    // ========== 导航 ==========
    function mmPrev() {
        if (mmIdx > 0) mmSelect(mmIdx - 1);
    }

    function mmNext() {
        if (mmIdx < mmImages.length - 1) mmSelect(mmIdx + 1);
    }

    // ========== 渲染 ==========
    function mmRender() {
        const canvas = document.getElementById('mm_canvas');
        if (!canvas || !mmImg || !mmLayers.auto) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        const iw = mmImg.naturalWidth, ih = mmImg.naturalHeight;
        
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(mmImg, 0, 0, w, h);
        
        const opacity = (document.getElementById('mm_overlay_opacity')?.value || 40) / 100;
        const showAuto = document.getElementById('mm_layer_auto')?.checked !== false;
        const showManual = document.getElementById('mm_layer_manual')?.checked !== false;
        const showInverse = document.getElementById('mm_layer_inverse')?.checked !== false;
        
        // 读取 mask 数据
        const autoData = mmLayers.auto.getContext('2d').getImageData(0, 0, iw, ih).data;
        const manualData = mmLayers.manual.getContext('2d').getImageData(0, 0, iw, ih).data;
        const inverseData = mmLayers.inverse.getContext('2d').getImageData(0, 0, iw, ih).data;
        
        // 创建叠加层
        if (!window._mmOverlayCanvas) window._mmOverlayCanvas = document.createElement('canvas');
        const oc = window._mmOverlayCanvas;
        oc.width = w; oc.height = h;
        const octx = oc.getContext('2d');
        const overlayData = octx.createImageData(w, h);
        const od = overlayData.data;
        const alpha255 = Math.round(opacity * 255);
        
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const mx = Math.min(Math.floor(x / mmScale), iw - 1);
                const my = Math.min(Math.floor(y / mmScale), ih - 1);
                const mi = (my * iw + mx) * 4;
                const oi = (y * w + x) * 4;
                const aVal = autoData[mi] > 127;
                const mVal = manualData[mi] > 127;
                const iVal = inverseData[mi] > 127;
                
                let r = 0, g = 0, b = 0, a = 0;
                if (showInverse && iVal) { r = 0; g = 255; b = 0; a = alpha255; }
                else if (showManual && mVal) { r = 255; g = 255; b = 0; a = alpha255; }
                else if (showAuto && aVal) { r = 255; g = 0; b = 0; a = alpha255; }
                
                od[oi] = r; od[oi + 1] = g; od[oi + 2] = b; od[oi + 3] = a;
            }
        }
        octx.putImageData(overlayData, 0, 0);
        ctx.drawImage(oc, 0, 0);
        
        // SAM2 预览层
        const showSam2 = document.getElementById('mm_layer_sam2')?.checked !== false;
        if (showSam2 && mmSam2PreviewLayer) {
            const sam2Data = mmSam2PreviewLayer.getContext('2d').getImageData(0, 0, mmSam2PreviewLayer.width, mmSam2PreviewLayer.height).data;
            if (!window._mmSam2OverlayCanvas) window._mmSam2OverlayCanvas = document.createElement('canvas');
            const soc = window._mmSam2OverlayCanvas;
            soc.width = w; soc.height = h;
            const soctx = soc.getContext('2d');
            const sam2Overlay = soctx.createImageData(w, h);
            const sod = sam2Overlay.data;
            const sam2Alpha = Math.round(alpha255 * 1.2);
            
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const mx = Math.min(Math.floor(x / mmScale), iw - 1);
                    const my = Math.min(Math.floor(y / mmScale), ih - 1);
                    const mi = (my * iw + mx) * 4;
                    const oi = (y * w + x) * 4;
                    if (sam2Data[mi] > 127) {
                        sod[oi] = 59; sod[oi + 1] = 130; sod[oi + 2] = 246; sod[oi + 3] = sam2Alpha;
                    }
                }
            }
            soctx.putImageData(sam2Overlay, 0, 0);
            ctx.drawImage(soc, 0, 0);
        }
        
        // 绘制 SAM2 点标记
        if (mmTool === 'sam2' && mmSam2Points.length > 0) {
            ctx.save();
            mmSam2Points.forEach((pt, i) => {
                const color = mmSam2Labels[i] === 1 ? '#44ff44' : '#ff4444';
                ctx.fillStyle = color;
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(pt[0] * mmScale, pt[1] * mmScale, 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            });
            ctx.restore();
        }
        
        // 绘制十字准线（鼠标悬停时显示）
        if (mmShowCrosshair && mmMouseX >= 0 && mmMouseY >= 0) {
            ctx.save();
            // 更鲜艳的亮青色，带黑色描边增强对比度
            ctx.strokeStyle = '#00FFFF';
            ctx.lineWidth = 2;
            ctx.setLineDash([]);
            ctx.shadowColor = '#000';
            ctx.shadowBlur = 3;
            
            // 水平线
            ctx.beginPath();
            ctx.moveTo(0, mmMouseY * mmScale);
            ctx.lineTo(w, mmMouseY * mmScale);
            ctx.stroke();
            
            // 垂直线
            ctx.beginPath();
            ctx.moveTo(mmMouseX * mmScale, 0);
            ctx.lineTo(mmMouseX * mmScale, h);
            ctx.stroke();
            
            // 中心点（大十字标记）
            ctx.shadowBlur = 0;
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 2;
            const cx = mmMouseX * mmScale, cy = mmMouseY * mmScale;
            ctx.beginPath();
            ctx.moveTo(cx - 8, cy);
            ctx.lineTo(cx + 8, cy);
            ctx.moveTo(cx, cy - 8);
            ctx.lineTo(cx, cy + 8);
            ctx.stroke();
            
            // 中心点填充
            ctx.fillStyle = '#00FFFF';
            ctx.beginPath();
            ctx.arc(cx, cy, 3, 0, Math.PI * 2);
            ctx.fill();
            
            // 坐标文字（带背景）
            ctx.shadowBlur = 0;
            const text = '[' + Math.round(mmMouseX) + ',' + Math.round(mmMouseY) + ']';
            ctx.font = 'bold 12px Arial';
            const textWidth = ctx.measureText(text).width;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(mmMouseX * mmScale + 10, mmMouseY * mmScale - 18, textWidth + 6, 16);
            ctx.fillStyle = '#00FFFF';
            ctx.fillText(text, mmMouseX * mmScale + 13, mmMouseY * mmScale - 6);
            ctx.restore();
        }
        
        // 绘制 SAM2 框选预览
        if (mmTool === 'sam2' && mmSam2Mode === 'bbox' && mmBboxState === 'waiting_end') {
            const x1 = Math.min(mmBboxStartX, mmBboxEndX);
            const y1 = Math.min(mmBboxStartY, mmBboxEndY);
            const x2 = Math.max(mmBboxStartX, mmBboxEndX);
            const y2 = Math.max(mmBboxStartY, mmBboxEndY);
            
            ctx.save();
            ctx.strokeStyle = mmSam2Positive ? '#44ff44' : '#ff4444';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(x1 * mmScale, y1 * mmScale, (x2 - x1) * mmScale, (y2 - y1) * mmScale);
            
            ctx.fillStyle = mmSam2Positive ? 'rgba(68, 255, 68, 0.15)' : 'rgba(255, 68, 68, 0.15)';
            ctx.fillRect(x1 * mmScale, y1 * mmScale, (x2 - x1) * mmScale, (y2 - y1) * mmScale);
            
            // 起点标记
            ctx.fillStyle = '#ffaa00';
            ctx.beginPath();
            ctx.arc(mmBboxStartX * mmScale, mmBboxStartY * mmScale, 5, 0, Math.PI * 2);
            ctx.fill();
            
            // 尺寸信息
            ctx.fillStyle = '#fff';
            ctx.font = '12px Arial';
            ctx.fillText(Math.round(x2 - x1) + ' x ' + Math.round(y2 - y1), x2 * mmScale + 5, y1 * mmScale + 15);
            ctx.restore();
        }
        
        // 套索预览
        if (mmTool === 'lasso' && mmLassoPoints.length > 0) {
            ctx.save();
            
            if (mmLassoPoints.length >= 3) {
                ctx.beginPath();
                ctx.moveTo(mmLassoPoints[0].x * mmScale, mmLassoPoints[0].y * mmScale);
                for (let i = 1; i < mmLassoPoints.length; i++) {
                    ctx.lineTo(mmLassoPoints[i].x * mmScale, mmLassoPoints[i].y * mmScale);
                }
                if (mmLassoMousePos) ctx.lineTo(mmLassoMousePos.x * mmScale, mmLassoMousePos.y * mmScale);
                ctx.closePath();
                ctx.fillStyle = 'rgba(255, 255, 0, 0.12)';
                ctx.fill();
            }
            
            if (mmLassoPoints.length >= 2) {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                ctx.moveTo(mmLassoPoints[0].x * mmScale, mmLassoPoints[0].y * mmScale);
                for (let i = 1; i < mmLassoPoints.length; i++) {
                    ctx.lineTo(mmLassoPoints[i].x * mmScale, mmLassoPoints[i].y * mmScale);
                }
                ctx.stroke();
            }
            
            if (mmLassoMousePos && mmLassoPoints.length >= 1) {
                ctx.strokeStyle = 'rgba(100, 180, 255, 0.7)';
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                const last = mmLassoPoints[mmLassoPoints.length - 1];
                ctx.moveTo(last.x * mmScale, last.y * mmScale);
                ctx.lineTo(mmLassoMousePos.x * mmScale, mmLassoMousePos.y * mmScale);
                ctx.stroke();
            }
            
            // 锚点
            mmLassoPoints.forEach((p, i) => {
                ctx.fillStyle = i === 0 ? '#ff6b6b' : '#69db7c';
                ctx.strokeStyle = 'rgba(0,0,0,0.6)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(p.x * mmScale, p.y * mmScale, 3, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            });
            ctx.restore();
        }
    }

    // ========== 导出全局函数 ==========
    window.mmLoadFolder = mmLoadFolder;
    window.mmSelect = mmSelect;
    window.mmSetTool = mmSetTool;
    window.mmSetTarget = mmSetTarget;
    window.mmSetBrush = mmSetBrush;
    window.mmSetFillTolerance = mmSetFillTolerance;
    window.mmSetZoom = mmSetZoom;
    window.mmUndo = mmUndo;
    window.mmReset = mmReset;
    window.mmSaveLayers = mmSaveLayers;
    window.mmMergePreview = mmMergePreview;
    window.mmMergeSave = mmMergeSave;
    window.mmSam2SetMode = mmSam2SetMode;
    window.mmSam2SetPositive = mmSam2SetPositive;
    window.mmSam2Confirm = mmSam2Confirm;
    window.mmSam2Clear = mmSam2Clear;
    window.mmSam2UpdatePanel = mmSam2UpdatePanel;
    window.mmPrev = mmPrev;
    window.mmNext = mmNext;

    // 初始化
    initManualMaskEditor();
})();
