/**
 * 高级 Mask 编辑器 - 完整前端实现
 * 支持三层 mask 系统 + SAM2 点击分割 + SAM2 矩形框选（点按式）+ 文件夹批量编辑
 * 框选模式：点按起点 → 移动预览 → 点按终点（类似 x-anylabeling）
 */

class AdvancedMaskEditor {
    constructor() {
        this.canvas = document.getElementById('editor-canvas');
        this.ctx = this.canvas.getContext('2d', {willReadFrequently: true});
        this.container = document.getElementById('canvas-container');
        
        // 图层数据
        this.layers = {
            image: null,      // 原图
            auto: null,       // 自动检测层（只读）
            manual: null,     // 手动添加层
            inverse: null,    // 反向保留层
            sam2: null        // SAM2 预览层
        };
        this.layerNames = ['auto', 'manual', 'inverse', 'sam2'];
        this.layerColors = {
            auto:   'rgba(255, 0,   0,   OPACITY)',
            manual: 'rgba(255, 255, 0,   OPACITY)',
            inverse:'rgba(0,   255, 0,   OPACITY)',
            sam2:   'rgba(0,   136, 255, OPACITY)'
        };
        
        // 编辑状态
        this.currentTool = 'add';
        this.brushSize = 20;
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        
        // 视图变换
        this.viewScale = 1.0;
        this.viewOffsetX = 0;
        this.viewOffsetY = 0;
        this.isPanning = false;
        this.panStartX = 0;
        this.panStartY = 0;
        this.panStartOffsetX = 0;
        this.panStartOffsetY = 0;
        this.isSpacePressed = false;
        
        // 历史记录
        this.history = [];
        this.historyIndex = -1;
        this.maxHistory = 50;
        this.previousTool = 'add';
        
        // 文件夹导航
        this.folder = null;
        this.images = [];
        this.currentIndex = -1;
        this.imagePath = null;
        
        // SAM2 点模式
        this.sam2Points = [];
        this.sam2Labels = [];
        this.sam2Loading = false;
        
        // SAM2 矩形框模式 - 点按式（类似 x-anylabeling）
        this.sam2BboxMode = false;       // 是否启用 bbox 模式
        this.sam2BboxPositive = true;     // true=正向选择(选取区域), false=反向选择(排除区域)
        
        // 点按式框选状态机: 'idle' | 'waiting_end'
        // 'idle': 空闲状态，等待点击确定起点
        // 'waiting_end': 已确定起点，移动鼠标显示预览框，等待点击确定终点
        this.bboxState = 'idle';
        this.bboxStartX = 0;
        this.bboxStartY = 0;
        this.bboxEndX = 0;
        this.bboxEndY = 0;
        
        // 十字准线 - 鼠标位置显示水平和垂直线
        this.crosshairX = -1;
        this.crosshairY = -1;
        this.showCrosshair = false;
        
        // 用于存储所有框选的结果（支持多框选）
        this.sam2BboxResults = [];
        
        this.init();
    }
    
    init() {
        this.setupEvents();
        this.updateUndoRedoButtons();
        
        // 尝试从 URL 参数加载图片
        const imagePath = new URLSearchParams(window.location.search).get('image');
        if (imagePath) {
            this.loadImageByPath(imagePath);
        }
    }
    
    // ========== 文件夹导航 ==========
    
    async loadFolder() {
        const input = document.getElementById('folder-path');
        const folder = input.value.trim();
        if (!folder) { alert('请输入文件夹路径'); return; }
        
        try {
            const res = await fetch('/editor/list_images', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({folder})
            });
            const data = await res.json();
            if (!data.success) { alert(data.error || '加载失败'); return; }
            
            this.folder = data.folder;
            this.images = data.images;
            this.currentIndex = -1;
            
            document.getElementById('nav-info').style.display = 'block';
            document.getElementById('nav-total').textContent = this.images.length;
            
            if (this.images.length > 0) {
                this.gotoImage(0);
            } else {
                alert('文件夹中没有图片');
            }
        } catch (e) {
            alert('加载文件夹失败: ' + e.message);
        }
    }
    
    gotoImage(idx) {
        if (!this.images.length) return;
        if (idx < 0) idx = 0;
        if (idx >= this.images.length) idx = this.images.length - 1;
        this.currentIndex = idx;
        
        const fname = this.images[idx].filename;
        this.imagePath = this.folder + '/' + fname;
        
        document.getElementById('nav-current').textContent = idx + 1;
        this.loadImageByPath(this.imagePath);
    }
    
    nextImage() {
        if (this.currentIndex < this.images.length - 1) {
            this.gotoImage(this.currentIndex + 1);
        }
    }
    
    prevImage() {
        if (this.currentIndex > 0) {
            this.gotoImage(this.currentIndex - 1);
        }
    }
    
    // ========== 图片加载 ==========
    
    async loadImageByPath(imagePath) {
        this.imagePath = imagePath;
        try {
            const data = await this.loadEditorData();
            await this.loadImage(data.image_url);
            await this.loadMask('auto',  data.auto_mask_url);
            await this.loadMask('manual', data.manual_mask_url);
            await this.loadMask('inverse', data.inverse_mask_url);
            
            // SAM2 层初始化为空
            this.layers.sam2 = this.createEmptyMask();
            
            // 清空 SAM2 点和框
            this.sam2Points = [];
            this.sam2Labels = [];
            this.sam2BboxResults = [];
            this.bboxState = 'idle';
            
            this.updateSam2Panel();
            
            this.setupCanvas();
            this.render();
            this.updateTransform();
            this.history = [];
            this.historyIndex = -1;
            this.updateUndoRedoButtons();
            
            console.log('加载图片:', imagePath);
        } catch (error) {
            console.error('加载失败:', error);
            alert('加载失败: ' + error.message);
        }
    }
    
    async loadEditorData() {
        const response = await fetch('/editor/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image_path: this.imagePath})
        });
        if (!response.ok) throw new Error('加载编辑器数据失败');
        const data = await response.json();
        return {
            image_url: `/editor/get_image?path=${encodeURIComponent(this.imagePath)}`,
            auto_mask_url: data.masks?.auto ? `/editor/get_mask?path=${encodeURIComponent(data.masks.auto)}` : null,
            manual_mask_url: data.masks?.manual ? `/editor/get_mask?path=${encodeURIComponent(data.masks.manual)}` : null,
            inverse_mask_url: data.masks?.inverse ? `/editor/get_mask?path=${encodeURIComponent(data.masks.inverse)}` : null
        };
    }
    
    async loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.layers.image = img;
                this.canvas.width = img.width;
                this.canvas.height = img.height;
                resolve();
            };
            img.onerror = () => reject(new Error('图片加载失败'));
            img.src = url;
        });
    }
    
    async loadMask(layerName, url) {
        if (!url) {
            this.layers[layerName] = this.createEmptyMask();
            return;
        }
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const c = document.createElement('canvas');
                c.width = this.canvas.width;
                c.height = this.canvas.height;
                const x = c.getContext('2d');
                x.drawImage(img, 0, 0);
                this.layers[layerName] = x.getImageData(0, 0, c.width, c.height);
                resolve();
            };
            img.onerror = () => {
                this.layers[layerName] = this.createEmptyMask();
                resolve();
            };
            img.src = url;
        });
    }
    
    createEmptyMask() {
        if (this.canvas.width === 0) return null;
        return this.ctx.createImageData(this.canvas.width, this.canvas.height);
    }
    
    // ========== 视图变换 ==========
    
    setupCanvas() {
        this.fitToContainer();
    }
    
    fitToContainer() {
        const rect = this.container.getBoundingClientRect();
        const padding = 20;
        const availW = rect.width - padding * 2;
        const availH = rect.height - padding * 2;
        const imgW = this.canvas.width;
        const imgH = this.canvas.height;
        if (imgW > 0 && imgH > 0) {
            this.viewScale = Math.min(availW / imgW, availH / imgH, 1);
        } else {
            this.viewScale = 1.0;
        }
        const scaledW = this.canvas.width * this.viewScale;
        const scaledH = this.canvas.height * this.viewScale;
        this.viewOffsetX = (rect.width - scaledW) / 2;
        this.viewOffsetY = (rect.height - scaledH) / 2;
        this.updateTransform();
    }
    
    screenToCanvas(screenX, screenY) {
        return {
            x: (screenX - this.viewOffsetX) / this.viewScale,
            y: (screenY - this.viewOffsetY) / this.viewScale
        };
    }
    
    updateTransform() {
        this.canvas.style.transform = `translate(${this.viewOffsetX}px, ${this.viewOffsetY}px) scale(${this.viewScale})`;
        const z = document.getElementById('zoom-info');
        if (z) z.textContent = `${Math.round(this.viewScale * 100)}%`;
    }
    
    zoomAt(screenX, screenY, factor) {
        const canvasX = (screenX - this.viewOffsetX) / this.viewScale;
        const canvasY = (screenY - this.viewOffsetY) / this.viewScale;
        this.viewScale = Math.max(0.1, Math.min(10, this.viewScale * factor));
        this.viewOffsetX = screenX - canvasX * this.viewScale;
        this.viewOffsetY = screenY - canvasY * this.viewScale;
        this.updateTransform();
    }
    
    resetView() {
        this.fitToContainer();
    }
    
    // ========== 事件处理 ==========
    
    setupEvents() {
        // 滚轮缩放
        this.container.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = this.container.getBoundingClientRect();
            this.zoomAt(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.1 : 0.9);
        }, {passive: false});
        
        // Space 平移
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                if (!this.isSpacePressed) {
                    this.isSpacePressed = true;
                    this.canvas.style.cursor = 'grab';
                }
            }
            // A/D 翻页
            if (e.key === 'a' || e.key === 'A') {
                if (!this.isDrawing && !this.sam2Loading && this.bboxState === 'idle') this.prevImage();
            }
            if (e.key === 'd' || e.key === 'D') {
                if (!this.isDrawing && !this.sam2Loading && this.bboxState === 'idle') this.nextImage();
            }
            // E 橡皮擦
            if (e.key === 'e' || e.key === 'E') {
                if (this.currentTool !== 'erase') {
                    this.previousTool = this.currentTool;
                    this.setTool('erase');
                }
            }
            // Ctrl+S / Alt+Q 保存
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this.saveLayers();
            }
            if (e.altKey && e.key === 'q') {
                e.preventDefault();
                this.saveLayers();
            }
            // Ctrl+X 保存并下一张
            if ((e.ctrlKey || e.metaKey) && e.key === 'x') {
                e.preventDefault();
                this.saveAndNext();
            }
            // ESC 取消当前正在绘制的框
            if (e.key === 'Escape') {
                if (this.bboxState === 'waiting_end') {
                    this.cancelBboxDrawing();
                }
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (e.code === 'Space') {
                this.isSpacePressed = false;
                this.canvas.style.cursor = 'crosshair';
            }
            if (e.key === 'e' || e.key === 'E') {
                if (this.currentTool === 'erase') {
                    this.setTool(this.previousTool);
                }
            }
        });
        
        // 鼠标事件
        this.container.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            const rect = this.container.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;
            
            if (this.isSpacePressed) {
                this.isPanning = true;
                this.panStartX = e.clientX;
                this.panStartY = e.clientY;
                this.panStartOffsetX = this.viewOffsetX;
                this.panStartOffsetY = this.viewOffsetY;
                this.canvas.style.cursor = 'grabbing';
            } else if (this.currentTool === 'sam2' || this.currentTool === 'sam2_bbox') {
                // SAM2 模式：区分点和框模式
                const pos = this.screenToCanvas(sx, sy);
                
                if (this.sam2BboxMode) {
                    // 点按式矩形框模式（类似 x-anylabeling）
                    if (this.bboxState === 'idle') {
                        // 空闲状态：Shift+左键切换正向/反向
                        if (e.shiftKey) {
                            this.sam2BboxPositive = !this.sam2BboxPositive;
                            this.updateBboxModeUI();
                            return;
                        }
                        // 左键点击确定起点
                        this.bboxStartX = pos.x;
                        this.bboxStartY = pos.y;
                        this.bboxEndX = pos.x;
                        this.bboxEndY = pos.y;
                        this.bboxState = 'waiting_end';
                        this.updateBboxModeUI();
                    } else if (this.bboxState === 'waiting_end') {
                        // 等待终点状态：点击确定终点，完成当前框
                        this.bboxEndX = pos.x;
                        this.bboxEndY = pos.y;
                        this.finalizeBbox();
                        // 完成后自动开始新框
                        this.bboxStartX = pos.x;
                        this.bboxStartY = pos.y;
                        this.bboxEndX = pos.x;
                        this.bboxEndY = pos.y;
                        // 保持 waiting_end 状态，继续绘制下一个框
                        this.bboxState = 'waiting_end';
                        this.updateBboxModeUI();
                    }
                } else {
                    // 点模式：普通点击添加前景点，Shift点击添加背景点
                    const label = e.shiftKey ? 0 : 1;
                    this.sam2AddPoint(pos.x, pos.y, label);
                }
            } else {
                this.startDrawing(sx, sy);
            }
        });
        
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;
            
            if (this.isPanning) {
                this.viewOffsetX = this.panStartOffsetX + (e.clientX - this.panStartX);
                this.viewOffsetY = this.panStartOffsetY + (e.clientY - this.panStartY);
                this.updateTransform();
            } else if (this.isDrawing) {
                this.draw(sx, sy);
            } else {
                // 所有工具模式下都更新十字准线位置（类似 x-anylabeling）
                const pos = this.screenToCanvas(sx, sy);
                this.crosshairX = pos.x;
                this.crosshairY = pos.y;
                this.showCrosshair = true;
                
                // 如果在 SAM2 框模式且等待终点状态，更新预览框终点
                if ((this.currentTool === 'sam2' || this.currentTool === 'sam2_bbox') && this.bboxState === 'waiting_end') {
                    this.bboxEndX = pos.x;
                    this.bboxEndY = pos.y;
                }
                this.render(); // 重新渲染以显示十字线和预览框
            }
        });
        
        this.container.addEventListener('mouseup', () => {
            if (this.isPanning) {
                this.isPanning = false;
                this.canvas.style.cursor = this.isSpacePressed ? 'grab' : 'crosshair';
            }
            if (this.isDrawing) {
                this.stopDrawing();
            }
            // 注意：框选改为点按式，不再在 mouseup 时触发
        });
        
        this.container.addEventListener('mouseleave', () => {
            if (this.isPanning) {
                this.isPanning = false;
                this.canvas.style.cursor = 'crosshair';
            }
            if (this.isDrawing) this.stopDrawing();
            // 鼠标离开时隐藏十字准线
            this.showCrosshair = false;
            this.render();
        });
        
        // 右键取消当前正在绘制的框
        this.container.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            if (this.bboxState === 'waiting_end') {
                this.cancelBboxDrawing();
            }
        });
        
        // 工具切换
        document.querySelectorAll('.tool-btn[data-tool]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setTool(e.target.dataset.tool);
            });
        });
        
        // 画笔大小
        const bs = document.getElementById('brush-size');
        if (bs) {
            bs.addEventListener('input', (e) => {
                this.brushSize = parseInt(e.target.value);
                document.getElementById('brush-size-value').textContent = e.target.value;
            });
        }
        
        // 图层可见性
        this.layerNames.forEach(name => {
            const cb = document.getElementById(`layer-${name}`);
            const op = document.getElementById(`opacity-${name}`);
            if (cb) cb.addEventListener('change', () => this.render());
            if (op) op.addEventListener('input', () => this.render());
        });
    }
    
    setTool(tool) {
        this.currentTool = tool;
        document.querySelectorAll('.tool-btn[data-tool]').forEach(b => b.classList.remove('active', 'sam2-active'));
        const btn = document.querySelector(`.tool-btn[data-tool="${tool}"]`);
        if (btn) {
            btn.classList.add('active');
            if (tool === 'sam2') btn.classList.add('sam2-active');
        }
        
        // 显示/隐藏 SAM2 面板
        const panel = document.getElementById('sam2-panel');
        if (panel) panel.classList.toggle('active', tool === 'sam2' || tool === 'sam2_bbox');
        
        // 更新框模式UI
        if (tool === 'sam2' || tool === 'sam2_bbox') {
            this.updateBboxModeUI();
        }
        // 注意：不再切换工具时重置 bboxState，只有主动取消或右键才会取消
    }
    
    updateToolButtons() {
        this.setTool(this.currentTool);
    }
    
    // ========== SAM2 点模式 ==========
    
    sam2AddPoint(x, y, label) {
        if (!this.imagePath || this.sam2Loading) return;
        this.sam2Points.push([Math.round(x), Math.round(y)]);
        this.sam2Labels.push(label);
        this.updateSam2Panel();
        this.render();
        
        // 自动触发推理（每次加点都重新推理）
        this.sam2Predict();
    }
    
    // ========== SAM2 矩形框模式（点按式） ==========
    
    cancelBboxDrawing() {
        // 取消当前正在绘制的框
        this.bboxState = 'idle';
        this.bboxStartX = 0;
        this.bboxStartY = 0;
        this.bboxEndX = 0;
        this.bboxEndY = 0;
        this.updateBboxModeUI();
        this.render();
    }
    
    updateBboxModeUI() {
        // 更新模式切换按钮的显示状态
        const btnPositive = document.getElementById('btn-sam2-positive');
        const btnNegative = document.getElementById('btn-sam2-negative');
        
        if (btnPositive && btnNegative) {
            if (this.sam2BboxMode) {
                // 框模式：更新按钮状态
                btnPositive.classList.toggle('active', this.sam2BboxPositive);
                btnNegative.classList.toggle('active', !this.sam2BboxPositive);
            } else {
                // 点模式：显示点模式状态
                btnPositive.classList.remove('active');
                btnNegative.classList.remove('active');
            }
        }
        
        // 更新提示文字 - 点按式交互说明
        const modeHint = document.getElementById('bbox-mode-hint');
        if (modeHint) {
            if (this.sam2BboxMode) {
                const modeText = this.sam2BboxPositive ? '正向选取' : '反向排除';
                if (this.bboxState === 'waiting_end') {
                    modeHint.innerHTML = `<span style="color:#ffaa44">■ 绘制中...</span> ${this.sam2BboxPositive ? '🟢正向' : '🔴反向'} | <span style="color:#fff">点击确定终点</span> | 右键/ESC取消`;
                } else {
                    modeHint.innerHTML = `${this.sam2BboxPositive ? '🟢正向' : '🔴反向'} | <span style="color:#fff">点击确定起点</span> | Shift切换正/反向`;
                }
            } else {
                modeHint.innerHTML = '🖱️ 点选: 左键=前景(绿) | Shift+左键=背景(红)';
            }
        }
        
        // 更新列表提示
        const list = document.getElementById('sam2-point-list');
        if (list) {
            if (this.bboxState === 'waiting_end') {
                const x1 = Math.round(Math.min(this.bboxStartX, this.bboxEndX));
                const y1 = Math.round(Math.min(this.bboxStartY, this.bboxEndY));
                const x2 = Math.round(Math.max(this.bboxStartX, this.bboxEndX));
                const y2 = Math.round(Math.max(this.bboxStartY, this.bboxEndY));
                list.innerHTML = `<span style="color:#ffaa44">■ 起点 [${Math.round(this.bboxStartX)}, ${Math.round(this.bboxStartY)}]</span><br>预览 [${x1},${y1}]-[${x2},${y2}]`;
            }
        }
    }
    
    toggleBboxMode() {
        this.sam2BboxMode = !this.sam2BboxMode;
        this.updateBboxModeUI();
        
        // 重置框选状态
        this.bboxState = 'idle';
        this.bboxStartX = 0;
        this.bboxStartY = 0;
        this.bboxEndX = 0;
        this.bboxEndY = 0;
        
        // 如果切换到框模式，清空点
        if (this.sam2BboxMode) {
            this.sam2Points = [];
            this.sam2Labels = [];
            this.updateSam2Panel();
        }
        
        // 更新按钮状态
        const btnBbox = document.getElementById('btn-sam2-bbox');
        if (btnBbox) {
            btnBbox.classList.toggle('active', this.sam2BboxMode);
        }
        
        this.render();
    }
    
    togglePositiveMode() {
        this.sam2BboxPositive = true;
        this.updateBboxModeUI();
    }
    
    toggleNegativeMode() {
        this.sam2BboxPositive = false;
        this.updateBboxModeUI();
    }
    
    finalizeBbox() {
        if (!this.imagePath || this.sam2Loading) return;
        
        // 确保矩形有足够的面积
        const dx = Math.abs(this.bboxEndX - this.bboxStartX);
        const dy = Math.abs(this.bboxEndY - this.bboxStartY);
        if (dx < 5 || dy < 5) {
            // 矩形太小，忽略
            return;
        }
        
        // 计算 bbox 坐标 (x1, y1, x2, y2)
        const x1 = Math.round(Math.min(this.bboxStartX, this.bboxEndX));
        const y1 = Math.round(Math.min(this.bboxStartY, this.bboxEndY));
        const x2 = Math.round(Math.max(this.bboxStartX, this.bboxEndX));
        const y2 = Math.round(Math.max(this.bboxStartY, this.bboxEndY));
        
        // 保存框选结果用于后续确认
        this.sam2BboxResults.push({
            bbox: [x1, y1, x2, y2],
            positive: this.sam2BboxPositive
        });
        
        // 立即执行推理
        this.sam2BboxPredict(x1, y1, x2, y2, this.sam2BboxPositive);
        
        // 重新渲染
        this.render();
    }
    
    async sam2BboxPredict(x1, y1, x2, y2, positive) {
        this.sam2Loading = true;
        
        const status = document.getElementById('sam2-status');
        if (status) status.innerHTML = '<span class="status-badge loading">框选推理中...</span>';
        
        const modelSelect = document.getElementById('sam2-model-select');
        const modelName = modelSelect ? modelSelect.value : null;
        
        try {
            const res = await fetch('/editor/sam2_bbox', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    image_path: this.imagePath,
                    bbox: JSON.stringify([x1, y1, x2, y2]),
                    model_name: modelName
                })
            });
            const data = await res.json();
            if (!data.success) {
                if (status) status.innerHTML = '<span class="status-badge error">失败</span>';
                console.error('SAM2 bbox 推理失败:', data.error);
                return;
            }
            
            // 解码 mask base64
            const img = new Image();
            img.onload = () => {
                const c = document.createElement('canvas');
                c.width = this.canvas.width;
                c.height = this.canvas.height;
                const x = c.getContext('2d');
                x.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
                const newMask = x.getImageData(0, 0, this.canvas.width, this.canvas.height);
                
                // 如果是反向模式，反转 mask
                if (!positive) {
                    const d = newMask.data;
                    for (let i = 0; i < d.length; i += 4) {
                        d[i] = d[i+1] = d[i+2] = 255 - d[i];
                    }
                }
                
                // 合并到 SAM2 预览层
                if (!this.layers.sam2) {
                    this.layers.sam2 = newMask;
                } else {
                    // 合并多个框选结果：正向用OR，反向用AND
                    this.mergeMasks(this.layers.sam2, newMask, positive);
                }
                
                // 显示 SAM2 预览层
                document.getElementById('sam2-preview-layer-item').style.display = 'block';
                const cb = document.getElementById('layer-sam2');
                if (cb && !cb.checked) cb.checked = true;
                
                const bboxCount = this.sam2BboxResults.length;
                if (status) status.innerHTML = `<span class="status-badge ok">完成 (${bboxCount}框)</span>`;
                this.updateSam2Panel();
                this.render();
            };
            img.src = data.mask_base64;
        } catch (e) {
            console.error('SAM2 bbox 请求失败:', e);
            if (status) status.innerHTML = '<span class="status-badge error">错误</span>';
        } finally {
            this.sam2Loading = false;
        }
    }
    
    mergeMasks(target, source, isPositive) {
        const td = target.data;
        const sd = source.data;
        
        if (isPositive) {
            // 正向：OR 合并
            for (let i = 0; i < td.length; i += 4) {
                if (sd[i] > 127) {
                    td[i] = td[i+1] = td[i+2] = 255;
                    td[i+3] = 255;
                }
            }
        } else {
            // 反向：反转后 AND 合并
            for (let i = 0; i < td.length; i += 4) {
                if (td[i] > 127 && sd[i] > 127) {
                    // 保留交叉部分
                    td[i] = td[i+1] = td[i+2] = 255;
                    td[i+3] = 255;
                } else {
                    // 清除非交叉部分
                    td[i] = td[i+1] = td[i+2] = 0;
                    td[i+3] = 255;
                }
            }
        }
    }
    
    // ========== SAM2 统一 ==========
    
    updateSam2Panel() {
        const list = document.getElementById('sam2-point-list');
        if (!list) return;
        
        let html = '';
        
        // 显示点模式
        if (this.sam2Points.length > 0) {
            html += '<div style="margin-bottom: 8px; font-weight:600;">点选模式:</div>';
            for (let i = 0; i < this.sam2Points.length; i++) {
                const cls = this.sam2Labels[i] === 1 ? 'positive' : 'negative';
                const text = this.sam2Labels[i] === 1 ? '前景' : '背景';
                html += `<div class="sam2-point-item">
                    <span class="sam2-point-dot ${cls}"></span>
                    [${this.sam2Points[i][0]}, ${this.sam2Points[i][1]}] ${text}
                </div>`;
            }
        }
        
        // 显示框模式
        if (this.sam2BboxResults.length > 0) {
            if (html) html += '<hr style="border-color:#505050;margin:8px 0;">';
            html += '<div style="margin-bottom: 8px; font-weight:600;">框选模式:</div>';
            for (let i = 0; i < this.sam2BboxResults.length; i++) {
                const bbox = this.sam2BboxResults[i].bbox;
                const type = this.sam2BboxResults[i].positive ? '正向' : '反向';
                const color = this.sam2BboxResults[i].positive ? '#44ff44' : '#ff4444';
                html += `<div class="sam2-point-item">
                    <span class="sam2-point-dot" style="background:${color}"></span>
                    [${bbox[0]},${bbox[1]}]-[${bbox[2]},${bbox[3]}] ${type}
                </div>`;
            }
        }
        
        if (!html) {
            html = this.sam2BboxMode 
                ? '点击画布确定矩形起点...' 
                : '点击画布添加点...';
        }
        
        // 如果正在绘制，更新绘制状态
        if (this.bboxState === 'waiting_end' && this.sam2BboxMode) {
            const x1 = Math.round(Math.min(this.bboxStartX, this.bboxEndX));
            const y1 = Math.round(Math.min(this.bboxStartY, this.bboxEndY));
            const x2 = Math.round(Math.max(this.bboxStartX, this.bboxEndX));
            const y2 = Math.round(Math.max(this.bboxStartY, this.bboxEndY));
            const color = this.sam2BboxPositive ? '#44ff44' : '#ff4444';
            html += `<hr style="border-color:#505050;margin:8px 0;">
                <div style="color:#ffaa44; font-weight:600;">■ 绘制中:</div>
                <div class="sam2-point-item">
                    <span class="sam2-point-dot" style="background:${color}"></span>
                    起点 [${Math.round(this.bboxStartX)}, ${Math.round(this.bboxStartY)}]
                </div>
                <div class="sam2-point-item">
                    预览 [${x1},${y1}]-[${x2},${y2}]
                </div>`;
        }
        
        list.innerHTML = html;
    }
    
    async sam2Predict() {
        if (this.sam2Points.length === 0) return;
        this.sam2Loading = true;
        
        const status = document.getElementById('sam2-status');
        if (status) status.innerHTML = '<span class="status-badge loading">推理中...</span>';
        
        const modelSelect = document.getElementById('sam2-model-select');
        const modelName = modelSelect ? modelSelect.value : null;
        
        try {
            const res = await fetch('/editor/sam2_point', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    image_path: this.imagePath,
                    points: JSON.stringify(this.sam2Points),
                    labels: JSON.stringify(this.sam2Labels),
                    model_name: modelName
                })
            });
            const data = await res.json();
            if (!data.success) {
                if (status) status.innerHTML = '<span class="status-badge error">失败</span>';
                console.error('SAM2 推理失败:', data.error);
                return;
            }
            
            // 解码 mask base64
            const img = new Image();
            img.onload = () => {
                const c = document.createElement('canvas');
                c.width = this.canvas.width;
                c.height = this.canvas.height;
                const x = c.getContext('2d');
                x.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
                this.layers.sam2 = x.getImageData(0, 0, this.canvas.width, this.canvas.height);
                
                // 显示 SAM2 预览层
                document.getElementById('sam2-preview-layer-item').style.display = 'block';
                const cb = document.getElementById('layer-sam2');
                if (cb && !cb.checked) cb.checked = true;
                
                if (status) status.innerHTML = `<span class="status-badge ok">完成 (${data.points_count}点)</span>`;
                this.render();
            };
            img.src = data.mask_base64;
        } catch (e) {
            console.error('SAM2 请求失败:', e);
            if (status) status.innerHTML = '<span class="status-badge error">错误</span>';
        } finally {
            this.sam2Loading = false;
        }
    }
    
    sam2Confirm() {
        if (!this.layers.sam2) return;
        // 将 SAM2 预览层合并到 manual 层
        this.saveHistory();
        this.mergeImageData(this.layers.manual, this.layers.sam2);
        this.sam2ClearPoints();
        this.render();
    }
    
    sam2Cancel() {
        this.sam2ClearPoints();
    }
    
    sam2ClearPoints() {
        this.sam2Points = [];
        this.sam2Labels = [];
        this.sam2BboxResults = [];
        this.bboxState = 'idle';
        this.bboxStartX = 0;
        this.bboxStartY = 0;
        this.bboxEndX = 0;
        this.bboxEndY = 0;
        this.layers.sam2 = this.createEmptyMask();
        document.getElementById('sam2-preview-layer-item').style.display = 'none';
        this.updateSam2Panel();
        this.updateBboxModeUI();
        const status = document.getElementById('sam2-status');
        if (status) status.textContent = '等待操作...';
        this.render();
    }
    
    mergeImageData(target, source) {
        const td = target.data;
        const sd = source.data;
        for (let i = 0; i < td.length; i += 4) {
            if (sd[i] > 127) {
                td[i] = td[i+1] = td[i+2] = 255;
                td[i+3] = 255;
            }
        }
    }
    
    // ========== 绘制逻辑 ==========
    
    startDrawing(screenX, screenY) {
        this.isDrawing = true;
        this.saveHistory();
        const pos = this.screenToCanvas(screenX, screenY);
        this.lastX = pos.x;
        this.lastY = pos.y;
        this.draw(screenX, screenY);
    }
    
    draw(screenX, screenY) {
        if (!this.isDrawing) return;
        const pos = this.screenToCanvas(screenX, screenY);
        let layer = null;
        if (this.currentTool === 'add') layer = 'manual';
        else if (this.currentTool === 'inverse') layer = 'inverse';
        else if (this.currentTool === 'erase') layer = 'manual';
        if (layer) {
            this.drawLine(this.layers[layer], this.lastX, this.lastY, pos.x, pos.y,
                this.brushSize, this.currentTool === 'erase' ? 0 : 255);
            this.lastX = pos.x;
            this.lastY = pos.y;
            this.render();
        }
    }
    
    stopDrawing() {
        this.isDrawing = false;
    }
    
    drawLine(imageData, x0, y0, x1, y1, radius, value) {
        const dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0);
        const sx = x0 < x1 ? 1 : -1;
        const sy = y0 < y1 ? 1 : -1;
        let err = dx - dy;
        while (true) {
            this.drawCircle(imageData, Math.floor(x0), Math.floor(y0), radius, value);
            if (Math.abs(x0 - x1) < 1 && Math.abs(y0 - y1) < 1) break;
            const e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }
    
    drawCircle(imageData, cx, cy, radius, value) {
        const data = imageData.data, w = imageData.width, h = imageData.height;
        for (let y = -radius; y <= radius; y++) {
            for (let x = -radius; x <= radius; x++) {
                if (x*x + y*y <= radius*radius) {
                    const px = Math.floor(cx + x), py = Math.floor(cy + y);
                    if (px >= 0 && px < w && py >= 0 && py < h) {
                        const idx = (py * w + px) * 4;
                        data[idx] = data[idx+1] = data[idx+2] = value;
                        data[idx+3] = 255;
                    }
                }
            }
        }
    }
    
    // ========== 渲染 ==========
    
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.layers.image) this.ctx.drawImage(this.layers.image, 0, 0);
        
        this.layerNames.forEach(name => this.renderLayer(name));
        
        // 绘制 SAM2 点标记
        if ((this.currentTool === 'sam2' || this.currentTool === 'sam2_bbox') && this.sam2Points.length > 0) {
            this.sam2Points.forEach((pt, i) => {
                const color = this.sam2Labels[i] === 1 ? '#44ff44' : '#ff4444';
                this.ctx.fillStyle = color;
                this.ctx.beginPath();
                this.ctx.arc(pt[0], pt[1], 6 / this.viewScale, 0, Math.PI * 2);
                this.ctx.fill();
                this.ctx.strokeStyle = '#000';
                this.ctx.lineWidth = 2 / this.viewScale;
                this.ctx.stroke();
            });
        }
        
        // 绘制十字准线（只要有图片加载就始终显示，类似 x-anylabeling）
        if (this.layers.image && this.showCrosshair && this.crosshairX >= 0) {
            this.drawCrosshair();
        }
        
        // 绘制正在绘制的预览框（点按式框选）
        if (this.bboxState === 'waiting_end') {
            const x1 = Math.min(this.bboxStartX, this.bboxEndX);
            const y1 = Math.min(this.bboxStartY, this.bboxEndY);
            const x2 = Math.max(this.bboxStartX, this.bboxEndX);
            const y2 = Math.max(this.bboxStartY, this.bboxEndY);
            
            // 绘制矩形框
            this.ctx.strokeStyle = this.sam2BboxPositive ? '#44ff44' : '#ff4444';
            this.ctx.lineWidth = 2 / this.viewScale;
            this.ctx.setLineDash([5 / this.viewScale, 5 / this.viewScale]);
            this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            this.ctx.setLineDash([]);
            
            // 填充半透明
            this.ctx.fillStyle = this.sam2BboxPositive 
                ? 'rgba(68, 255, 68, 0.15)' 
                : 'rgba(255, 68, 68, 0.15)';
            this.ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
            
            // 绘制起点标记
            this.ctx.fillStyle = '#ffaa44';
            this.ctx.beginPath();
            this.ctx.arc(this.bboxStartX, this.bboxStartY, 5 / this.viewScale, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.strokeStyle = '#fff';
            this.ctx.lineWidth = 1.5 / this.viewScale;
            this.ctx.stroke();
            
            // 显示尺寸信息
            const width = Math.round(x2 - x1);
            const height = Math.round(y2 - y1);
            this.ctx.fillStyle = '#fff';
            this.ctx.font = `${12 / this.viewScale}px Arial`;
            this.ctx.fillText(`${width} × ${height}`, x2 + 5 / this.viewScale, y1 + 15 / this.viewScale);
        }
        
        // 绘制已完成的框选区域标记
        if (this.sam2BboxResults.length > 0) {
            this.sam2BboxResults.forEach(bboxResult => {
                const [x1, y1, x2, y2] = bboxResult.bbox;
                this.ctx.strokeStyle = bboxResult.positive ? '#44ff44' : '#ff4444';
                this.ctx.lineWidth = 1.5 / this.viewScale;
                this.ctx.setLineDash([3 / this.viewScale, 3 / this.viewScale]);
                this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                this.ctx.setLineDash([]);
            });
        }
    }
    
    // 绘制十字准线
    drawCrosshair() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const x = this.crosshairX;
        const y = this.crosshairY;
        
        // 水平线
        this.ctx.strokeStyle = 'rgba(255, 170, 0, 0.6)';
        this.ctx.lineWidth = 1 / this.viewScale;
        this.ctx.setLineDash([]);
        this.ctx.beginPath();
        this.ctx.moveTo(0, y);
        this.ctx.lineTo(w, y);
        this.ctx.stroke();
        
        // 垂直线
        this.ctx.beginPath();
        this.ctx.moveTo(x, 0);
        this.ctx.lineTo(x, h);
        this.ctx.stroke();
        
        // 中心点标记
        this.ctx.fillStyle = '#ffaa00';
        this.ctx.beginPath();
        this.ctx.arc(x, y, 4 / this.viewScale, 0, Math.PI * 2);
        this.ctx.fill();
        
        // 显示坐标
        this.ctx.fillStyle = '#fff';
        this.ctx.font = `${11 / this.viewScale}px Arial`;
        const coordText = `[${Math.round(x)}, ${Math.round(y)}]`;
        this.ctx.fillText(coordText, x + 8 / this.viewScale, y - 8 / this.viewScale);
    }
    
    renderLayer(layerName) {
        const checkbox = document.getElementById(`layer-${layerName}`);
        if (!checkbox || !checkbox.checked) return;
        const slider = document.getElementById(`opacity-${layerName}`);
        const opacity = slider ? slider.value / 100 : 0.5;
        const imageData = this.layers[layerName];
        if (!imageData) return;
        
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.canvas.width;
        tempCanvas.height = this.canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        const imgData = tempCtx.createImageData(this.canvas.width, this.canvas.height);
        const nd = imgData.data, od = imageData.data;
        
        for (let i = 0; i < od.length; i += 4) {
            if (od[i] > 0) {
                nd[i] = nd[i+1] = nd[i+2] = 255;
                nd[i+3] = 255;
            }
        }
        tempCtx.putImageData(imgData, 0, 0);
        
        this.ctx.save();
        this.ctx.globalAlpha = opacity;
        const color = this.layerColors[layerName].replace('OPACITY', '1');
        this.ctx.fillStyle = color;
        this.ctx.globalCompositeOperation = 'multiply';
        this.ctx.drawImage(tempCanvas, 0, 0);
        this.ctx.restore();
    }
    
    // ========== 保存/合并 ==========
    
    async saveLayers() {
        if (!this.imagePath) { alert('请先加载图片'); return; }
        try {
            await this.saveLayer('manual');
            await this.saveLayer('inverse');
            alert('保存成功！');
        } catch (error) {
            console.error('保存失败:', error);
            alert('保存失败: ' + error.message);
        }
    }
    
    async saveLayer(layerName) {
        const base64 = this.imageDataToBase64(this.layers[layerName]);
        const res = await fetch('/editor/save_layer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                image_path: this.imagePath,
                layer_type: layerName,
                mask_data: base64
            })
        });
        if (!res.ok) throw new Error(`保存 ${layerName} 失败`);
    }
    
    async mergeLayers() {
        if (!this.imagePath) { alert('请先加载图片'); return; }
        try {
            const res = await fetch('/editor/merge', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    image_path: this.imagePath,
                    mode: 'standard'
                })
            });
            if (!res.ok) throw new Error('合并失败');
            const data = await res.json();
            if (data.preview_url) window.open(data.preview_url, '_blank');
            alert('合并完成！');
        } catch (error) {
            alert('合并失败: ' + error.message);
        }
    }
    
    async saveAndNext() {
        if (!this.imagePath) return;
        await this.saveLayers();
        if (this.currentIndex < this.images.length - 1) {
            this.nextImage();
        } else {
            alert('已经是最后一张了');
        }
    }
    
    imageDataToBase64(imageData) {
        const c = document.createElement('canvas');
        c.width = imageData.width;
        c.height = imageData.height;
        c.getContext('2d').putImageData(imageData, 0, 0);
        return c.toDataURL('image/png').split(',')[1];
    }
    
    // ========== 历史记录 ==========
    
    saveHistory() {
        this.history = this.history.slice(0, this.historyIndex + 1);
        this.history.push({
            manual: this.cloneImageData(this.layers.manual),
            inverse: this.cloneImageData(this.layers.inverse)
        });
        this.historyIndex++;
        if (this.history.length > this.maxHistory) {
            this.history.shift();
            this.historyIndex--;
        }
        this.updateUndoRedoButtons();
    }
    
    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            const s = this.history[this.historyIndex];
            this.layers.manual = this.cloneImageData(s.manual);
            this.layers.inverse = this.cloneImageData(s.inverse);
            this.render();
            this.updateUndoRedoButtons();
        }
    }
    
    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            const s = this.history[this.historyIndex];
            this.layers.manual = this.cloneImageData(s.manual);
            this.layers.inverse = this.cloneImageData(s.inverse);
            this.render();
            this.updateUndoRedoButtons();
        }
    }
    
    updateUndoRedoButtons() {
        const undoBtn = document.querySelector('[onclick="editor.undo()"]');
        const redoBtn = document.querySelector('[onclick="editor.redo()"]');
        if (undoBtn) undoBtn.disabled = this.historyIndex <= 0;
        if (redoBtn) redoBtn.disabled = this.historyIndex >= this.history.length - 1;
    }
    
    clearLayer(layerName) {
        if (confirm(`确定要清空${layerName === 'manual' ? '手动添加' : '反向保留'}层吗？`)) {
            this.saveHistory();
            this.layers[layerName] = this.createEmptyMask();
            this.render();
        }
    }
    
    cloneImageData(imageData) {
        return new ImageData(new Uint8ClampedArray(imageData.data), imageData.width, imageData.height);
    }
}

// 全局实例
let editor = null;
function initEditor() {
    editor = new AdvancedMaskEditor();
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initEditor);
} else {
    initEditor();
}
