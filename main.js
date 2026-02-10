// 主JavaScript文件 - 负责加载和显示视频

document.addEventListener('DOMContentLoaded', function() {
    loadComparisonVideos();
    wireContactButton();
});

// 加载对比视频
function loadComparisonVideos() {
    const t2vContainer = document.getElementById('comparison-t2v-container');
    const i2vContainer = document.getElementById('comparison-i2v-container');

    const { t2v, i2v } = getComparisonSets();

    if (t2vContainer) {
        if (!t2v || t2v.length === 0) {
            t2vContainer.innerHTML = '<div class="loading">No T2V comparisons yet. Add entries in videos-config.js.</div>';
        } else {
            t2vContainer.innerHTML = '';
            t2v.forEach(c => t2vContainer.appendChild(createComparisonGroup(c)));
        }
    }

    if (i2vContainer) {
        if (!i2v || i2v.length === 0) {
            i2vContainer.innerHTML = '<div class="loading">No I2V comparisons yet. Add entries in videos-config.js.</div>';
        } else {
            i2vContainer.innerHTML = '';
            i2v.forEach(c => i2vContainer.appendChild(createComparisonGroup(c)));
        }
    }
}

function getComparisonSets() {
    // Preferred format:
    // videosConfig.comparisonsT2V = [...]
    // videosConfig.comparisonsI2V = [...]
    if (videosConfig && Array.isArray(videosConfig.comparisonsT2V) && Array.isArray(videosConfig.comparisonsI2V)) {
        return { t2v: videosConfig.comparisonsT2V, i2v: videosConfig.comparisonsI2V };
    }

    // Backward compatible:
    // videosConfig.comparisons = [{ type: 't2v'|'i2v', ... }, ...]
    if (videosConfig && Array.isArray(videosConfig.comparisons)) {
        const all = videosConfig.comparisons;
        const t2v = all.filter(c => (c.type || '').toLowerCase() === 't2v' || /^comparison\s*t2v/i.test(c.title || ''));
        const i2v = all.filter(c => (c.type || '').toLowerCase() === 'i2v' || /^comparison\s*i2v/i.test(c.title || ''));
        const unknown = all.filter(c => !t2v.includes(c) && !i2v.includes(c));
        return { t2v: t2v.length ? t2v : unknown, i2v };
    }

    return { t2v: [], i2v: [] };
}

// 创建对比组
function createComparisonGroup(comparison) {
    const group = document.createElement('div');
    group.className = 'comparison-group';
    
    const rawTitle = (comparison && comparison.title ? String(comparison.title) : '').trim();
    const isGenericTitle = /^comparison\s+(t2v|i2v)\s+case\s+\d+/i.test(rawTitle) || /^comparison\s+case\s+\d+/i.test(rawTitle);
    const shouldShowTitle = rawTitle.length > 0 && !isGenericTitle;

    const title = document.createElement('div');
    title.className = 'comparison-title';
    title.textContent = rawTitle;
    
    const prompt = document.createElement('div');
    prompt.className = 'comparison-prompt';
    prompt.textContent = `Prompt: "${comparison.prompt}"`;
    
    const videosContainer = document.createElement('div');
    videosContainer.className = 'comparison-videos';
    
    // Render three methods in a fixed order.
    const methods = ['baseline', 'dancegrpo', 'ours'];
    const methodLabels = {
        'baseline': 'Baseline',
        'dancegrpo': 'DanceGRPO',
        'ours': 'TeleBoost'
    };
    
    const videoElements = [];

    methods.forEach(method => {
        const src = getComparisonVideoSrc(comparison, method);
        if (src) {
            const item = document.createElement('div');
            item.className = 'comparison-item';
            
            const video = document.createElement('video');
            video.src = src;
            video.controls = true;
            video.loop = true;
            video.muted = true;
            video.playsInline = true;
            video.preload = 'metadata';
            video.dataset.method = method;
            
            // 添加错误处理
            video.onerror = function() {
                video.style.display = 'none';
                const errorDiv = document.createElement('div');
                errorDiv.style.cssText = 'padding: 20px; background: #fee2e2; border-radius: 8px; color: #991b1b;';
                errorDiv.textContent = 'Failed to load video: ' + src;
                item.appendChild(errorDiv);
            };
            
            const label = document.createElement('div');
            label.className = `comparison-label ${method}`;
            label.textContent = methodLabels[method];
            
            item.appendChild(video);
            item.appendChild(label);
            videosContainer.appendChild(item);
            videoElements.push(video);
        }
    });
    
    if (shouldShowTitle) {
        group.appendChild(title);
    }
    group.appendChild(prompt);
    group.appendChild(videosContainer);

    // Keep the three method videos in sync within this comparison case.
    // This avoids the common “mismatch” feeling when users press play on one.
    wireSynchronizedPlayback(videoElements);
    
    return group;
}

function getComparisonVideoSrc(comparison, method) {
    const vids = comparison && comparison.videos ? comparison.videos : null;
    if (!vids) return null;

    // Accept various key casing / synonyms to avoid config footguns.
    const candidates = [
        method,
        method.toLowerCase(),
        method.toUpperCase(),
        method[0].toUpperCase() + method.slice(1),
    ];

    if (method === 'ours') {
        candidates.push('Ours', 'teleboost', 'TeleBoost');
    }

    for (const key of candidates) {
        if (vids[key]) return vids[key];
    }
    return null;
}

function wireSynchronizedPlayback(videos) {
    if (!videos || videos.length < 2) return;

    let isSyncing = false;

    const safePlay = (v) => {
        try {
            const p = v.play();
            if (p && typeof p.catch === 'function') p.catch(() => {});
        } catch (_) {}
    };

    const syncTimeFrom = (source) => {
        const t = Number.isFinite(source.currentTime) ? source.currentTime : 0;
        videos.forEach(v => {
            if (v === source) return;
            // Avoid excessive seeking jitter; only seek when meaningfully different.
            if (!Number.isFinite(v.currentTime) || Math.abs(v.currentTime - t) > 0.06) {
                try { v.currentTime = t; } catch (_) {}
            }
        });
    };

    videos.forEach(source => {
        source.addEventListener('play', () => {
            if (isSyncing) return;
            isSyncing = true;
            syncTimeFrom(source);
            videos.forEach(v => { if (v !== source) safePlay(v); });
            isSyncing = false;
        });

        source.addEventListener('pause', () => {
            if (isSyncing) return;
            isSyncing = true;
            videos.forEach(v => { if (v !== source) v.pause(); });
            isSyncing = false;
        });

        source.addEventListener('seeking', () => {
            if (isSyncing) return;
            isSyncing = true;
            syncTimeFrom(source);
            isSyncing = false;
        });

        source.addEventListener('ratechange', () => {
            if (isSyncing) return;
            isSyncing = true;
            const r = source.playbackRate;
            videos.forEach(v => { if (v !== source) v.playbackRate = r; });
            isSyncing = false;
        });
    });
}

function wireContactButton() {
    const btn = document.getElementById('contact-btn');
    if (!btn) return;

    const email = btn.getAttribute('data-email') || btn.getAttribute('href')?.replace(/^mailto:/i, '') || '';
    btn.addEventListener('click', async () => {
        if (!email) return;
        // Always provide a visible result even if mailto isn't configured.
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(email);
                showToast(`Copied: ${email}`);
            } else {
                showToast(`Email: ${email}`);
            }
        } catch (_) {
            showToast(`Email: ${email}`);
        }
    });
}

let toastTimer = null;
function showToast(message) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = message;
    el.classList.add('show');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 2000);
}

// 工具函数：验证视频文件是否存在（可选，需要服务器支持）
async function checkVideoExists(url) {
    try {
        const response = await fetch(url, { method: 'HEAD' });
        return response.ok;
    } catch (error) {
        return false;
    }
}
