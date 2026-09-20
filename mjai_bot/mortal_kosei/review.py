#!/usr/bin/env python3
"""Replay an mjai log through a self-trained Mortal model and emit an HTML
review report highlighting bad discards/decisions for one seat."""
import argparse
import gzip
import html
import json
import math
import os
import sys

DISCARD_TILES = (
    [f'{n}m' for n in range(1, 10)] +
    [f'{n}p' for n in range(1, 10)] +
    [f'{n}s' for n in range(1, 10)] +
    ['E', 'S', 'W', 'N', 'P', 'F', 'C'] +
    ['5mr', '5pr', '5sr']
)
ACTION_LABELS = {
    37: 'リーチ', 38: 'チー(下)', 39: 'チー(中)', 40: 'チー(上)',
    41: 'ポン', 42: 'カン', 43: '和了', 44: '流局', 45: 'スキップ',
}
HONOR_KANJI = {'E': '東', 'S': '南', 'W': '西', 'N': '北', 'P': '白', 'F': '發', 'C': '中'}
BAKAZE_KANJI = {'E': '東', 'S': '南', 'W': '西', 'N': '北'}
ACTOR_ACTION_TYPES = {'dahai', 'reach', 'chi', 'pon', 'ankan', 'kakan', 'daiminkan', 'hora', 'none'}


def candidate_meaning(idx):
    if idx < 37:
        return {'kind': 'discard', 'tile': DISCARD_TILES[idx], 'label': None}
    return {'kind': 'other', 'tile': None, 'label': ACTION_LABELS[idx]}


def tile_num(tile):
    if len(tile) < 2 or tile[0] not in '123456789':
        return None
    return int(tile[0])


def chi_variant_index(ev):
    pai_num = tile_num(ev['pai'])
    consumed_nums = [tile_num(t) for t in ev['consumed']]
    if pai_num is None or None in consumed_nums:
        return None
    nums = sorted(consumed_nums + [pai_num])
    pos = nums.index(pai_num)
    return {0: 38, 1: 39, 2: 40}[pos]


def action_index_for_event(ev):
    t = ev['type']
    if t == 'dahai':
        pai = ev['pai']
        return DISCARD_TILES.index(pai) if pai in DISCARD_TILES else None
    if t == 'reach':
        return 37
    if t == 'chi':
        return chi_variant_index(ev)
    if t == 'pon':
        return 41
    if t in ('ankan', 'kakan', 'daiminkan'):
        return 42
    if t == 'hora':
        return 43
    if t == 'ryukyoku':
        return 44
    if t == 'none':
        return 45
    return None


def kyoku_label(bakaze, kyoku, honba):
    label = f'{BAKAZE_KANJI.get(bakaze, bakaze)}{kyoku}局'
    if honba:
        label += f'{honba}本場'
    return label


def severity_of(rank, loss):
    if rank == 1:
        return 'agree'
    if loss < 20:
        return 'minor'
    if loss < 60:
        return 'notable'
    return 'major'


def stakes_of(shanten, junme):
    if shanten <= 1:
        stakes = '高'
    elif shanten == 2:
        stakes = '中'
    else:
        stakes = '低'
    if junme >= 15 and shanten >= 2:
        stakes = '低'
    return stakes


def read_lines(path):
    with open(path, 'rb') as f:
        head = f.read(2)
    opener = gzip.open if head == b'\x1f\x8b' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [ln for ln in (l.strip() for l in f) if ln]


def build_engine(bot_dir):
    import torch
    from model import Brain, DQN
    from engine import MortalEngine
    torch.set_num_threads(4)
    state = torch.load(os.path.join(bot_dir, 'mortal_final.pth'), weights_only=True, map_location='cpu')
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    mortal = Brain(version=version, num_blocks=cfg['resnet']['num_blocks'], conv_channels=cfg['resnet']['conv_channels']).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    return MortalEngine(
        mortal, dqn, version=version, is_oracle=False, device=torch.device('cpu'),
        enable_amp=False, enable_quick_eval=False, enable_rule_based_agari_guard=True, name='mortal',
    )


def tehai_to_tiles(tehai, akas):
    suits = ['m', 'p', 's']
    tiles = []
    for i, count in enumerate(tehai):
        if count == 0:
            continue
        if i < 27:
            suit = suits[i // 9]
            num = i % 9 + 1
            for _ in range(count):
                tiles.append(f'{num}{suit}')
        else:
            honor = ['E', 'S', 'W', 'N', 'P', 'F', 'C'][i - 27]
            for _ in range(count):
                tiles.append(honor)
    for suit_idx, has_aka in enumerate(akas):
        if not has_aka:
            continue
        suit = suits[suit_idx]
        for j, t in enumerate(tiles):
            if t == f'5{suit}':
                tiles[j] = f'5{suit}r'
                break
    return tiles


def replay(lines, seat, bot_dir):
    sys.path.insert(0, bot_dir)
    import torch
    from libriichi.mjai import Bot
    from libriichi.state import PlayerState

    engine = build_engine(bot_dir)
    bot = Bot(engine, seat)
    ps = PlayerState(seat)

    bakaze, kyoku, honba = 'E', 1, 0
    kyoku_index = -1
    kyoku_count = 0
    unmatched = 0

    decisions = []
    with torch.inference_mode():
        for i, line in enumerate(lines):
            ev = json.loads(line)
            if ev['type'] == 'start_kyoku':
                bakaze, kyoku, honba = ev['bakaze'], ev['kyoku'], ev['honba']
                kyoku_index += 1
                kyoku_count += 1

            reaction = bot.react(line)
            ps.update(line)

            if reaction is None or i + 1 >= len(lines):
                continue
            nxt = json.loads(lines[i + 1])
            if nxt['type'] not in ACTOR_ACTION_TYPES or nxt.get('actor') != seat:
                continue

            meta = json.loads(reaction).get('meta')
            if not meta or 'mask_bits' not in meta or 'q_values' not in meta:
                continue
            mask_bits = meta['mask_bits']
            q_values = meta['q_values']
            bits = [b for b in range(46) if mask_bits & (1 << b)]
            if len(bits) != len(q_values):
                unmatched += 1
                continue

            qs = torch.tensor(q_values, dtype=torch.float64)
            probs = torch.softmax(qs, dim=0).tolist()
            candidates = []
            for idx, q, p in zip(bits, q_values, probs):
                m = candidate_meaning(idx)
                candidates.append({'idx': idx, 'tile': m['tile'], 'label': m['label'], 'prob': p, 'q': q})
            candidates.sort(key=lambda c: -c['prob'])

            action_idx = action_index_for_event(nxt)
            legal_ranked = [c['idx'] for c in candidates]
            if action_idx is None or action_idx not in legal_ranked:
                unmatched += 1
                continue

            rank = legal_ranked.index(action_idx) + 1
            prob_mine = next(c['prob'] for c in candidates if c['idx'] == action_idx)
            prob_best = candidates[0]['prob']
            q_mine = next(c['q'] for c in candidates if c['idx'] == action_idx)
            loss = (prob_best - prob_mine) * 100
            sev = severity_of(rank, loss)
            junme = ps.at_turn
            shanten = ps.shanten

            mine_meaning = candidate_meaning(action_idx)
            decisions.append({
                'order': len(decisions),
                'kyoku_index': kyoku_index,
                'bakaze': bakaze, 'kyoku': kyoku, 'honba': honba,
                'label': kyoku_label(bakaze, kyoku, honba),
                'junme': junme,
                'shanten': shanten,
                'stakes': stakes_of(shanten, junme),
                'hand': tehai_to_tiles(list(ps.tehai), list(ps.akas_in_hand)),
                'has_meld': bool(len(ps.pons) or len(ps.chis) or len(ps.minkans) or len(ps.ankans)),
                'mine_tile': mine_meaning['tile'], 'mine_label': mine_meaning['label'],
                'candidates': candidates[:3],
                'rank': rank, 'prob_mine': prob_mine, 'prob_best': prob_best, 'q_mine': q_mine,
                'loss': loss, 'severity': sev,
            })

    return decisions, kyoku_count, unmatched


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
:root {
  --surface: #fcfcfb; --text: #0b0b0b; --text-sub: #52514e; --line: #2a78d6;
  --agree: #0ca30c; --minor: #fab219; --notable: #ec835a; --major: #d03b3b;
  --border: #dedcd4; --card-bg: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface: #1a1a19; --text: #ffffff; --text-sub: #c3c2b7; --line: #3987e5;
    --border: #3a3a37; --card-bg: #232320;
  }
}
:root[data-theme="dark"] {
  --surface: #1a1a19; --text: #ffffff; --text-sub: #c3c2b7; --line: #3987e5;
  --border: #3a3a37; --card-bg: #232320;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 16px; background: var(--surface); color: var(--text);
  font-family: -apple-system, "Hiragino Sans", "Yu Gothic", sans-serif;
  max-width: 980px; margin-left: auto; margin-right: auto;
}
h1 { font-size: 1.3em; margin-bottom: 4px; }
h2 { font-size: 1.05em; margin: 28px 0 8px; }
.header-meta { color: var(--text-sub); font-size: 0.9em; }
.tiles-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 14px 0; }
.stat-tile {
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 16px; min-width: 140px; flex: 1 1 140px;
}
.stat-tile .num { font-size: 1.5em; font-weight: 700; }
.stat-tile .lbl { font-size: 0.8em; color: var(--text-sub); }
.caption { font-size: 0.85em; color: var(--text-sub); margin-top: 4px; }
svg text { fill: var(--text-sub); font-size: 11px; }
.chart-wrap { position: relative; }
.tooltip {
  position: absolute; background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 10px; font-size: 0.8em; pointer-events: none;
  display: none; white-space: nowrap; z-index: 5;
}
.sort-btn {
  background: var(--card-bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 0.85em; margin-bottom: 10px;
}
.card {
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px; margin-bottom: 8px;
}
.card.collapsed { padding: 6px 14px; display: flex; align-items: center; gap: 10px; }
.card.collapsed .card-detail { display: none; }
.chip {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 26px; height: 30px; padding: 0 3px; border-radius: 5px;
  background: var(--surface); border: 1px solid var(--border); color: var(--text);
  font-size: 0.72em; margin: 2px; box-shadow: none;
}
.chip.ring { border-width: 2px; }
.severity-chip {
  display: inline-flex; align-items: center; gap: 4px; border-radius: 5px;
  padding: 2px 8px; font-size: 0.8em; font-weight: 600; color: var(--text);
  background: var(--card-bg); border: 2px solid var(--border);
}
.cand-line { font-size: 0.85em; margin: 2px 0; display: flex; align-items: center; gap: 4px; }
.cand-line.mine { font-weight: 700; }
.card-title { font-size: 0.85em; color: var(--text-sub); margin-bottom: 4px; }
footer { color: var(--text-sub); font-size: 0.78em; margin-top: 30px; line-height: 1.6; }
.grid-line { stroke: var(--border); stroke-width: 1; }
.series-line { stroke: var(--line); stroke-width: 2; fill: none; }
.crosshair { stroke: var(--text-sub); stroke-width: 1; display: none; }
.sev-agree { fill: var(--agree); }
.sev-minor { fill: var(--minor); }
.sev-notable { fill: var(--notable); }
.sev-major { fill: var(--major); }
.bar-fill { fill: var(--line); }
.stakes-badge {
  display: inline-flex; align-items: center; border-radius: 4px;
  padding: 1px 7px; font-size: 0.78em; color: var(--text-sub);
  border: 1px solid var(--border); background: var(--surface);
}
.stat-tile .sub { font-size: 0.75em; color: var(--text-sub); margin-top: 2px; }
</style>
</head>
<body>
<h1 id="title"></h1>
<div class="header-meta" id="header-meta"></div>
<div class="tiles-row" id="stat-tiles"></div>

<h2>累積 AIとの相違度</h2>
<div class="caption" id="cum-caption"></div>
<div class="chart-wrap"><svg id="cum-chart" width="100%" height="220"></svg><div class="tooltip" id="cum-tooltip"></div></div>

<h2>局ごとの相違度</h2>
<div class="chart-wrap"><svg id="bar-chart" width="100%" height="400"></svg><div class="tooltip" id="bar-tooltip"></div></div>

<h2>打牌ごとの判断</h2>
<button class="sort-btn" id="sort-toggle">並び順: 相違度順（切替）</button>
<div id="decision-list"></div>

<footer id="footnote"></footer>

<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const SEV_META = {
  agree: {icon: '◎', label: '一致', color: 'var(--agree)'},
  minor: {icon: '○', label: '小差', color: 'var(--minor)'},
  notable: {icon: '△', label: '相違', color: 'var(--notable)'},
  major: {icon: '✕', label: '大相違', color: 'var(--major)'},
};
const HONOR_KANJI = {E:'東', S:'南', W:'西', N:'北', P:'白', F:'發', C:'中'};

function tileLabel(t) {
  if (!t) return '';
  if (HONOR_KANJI[t]) return HONOR_KANJI[t];
  if (t.endsWith('r')) return t.slice(0, -1) + '赤';
  return t;
}
function candText(c) {
  return c.tile ? '打' + tileLabel(c.tile) : c.label;
}
function candDisplay(c) {
  return c.tile ? chipHtml(c.tile) : c.label;
}

document.title = DATA.title;
document.getElementById('title').textContent = DATA.title;
document.getElementById('header-meta').textContent =
  `自分の座席: ${DATA.seat} / 総局数: ${DATA.kyoku_count} / 検討した判断数: ${DATA.decisions.length}`;

const matched = DATA.decisions;
const agreeCount = matched.filter(d => d.severity === 'agree').length;
const totalLoss = matched.reduce((s, d) => s + d.loss, 0);
const majorCount = matched.filter(d => d.severity === 'major').length;
const majorHighStakesCount = matched.filter(d => d.severity === 'major' && d.stakes === '高').length;
let worst = matched[0];
for (const d of matched) if (d.loss > worst.loss) worst = d;

const tiles = [
  {num: matched.length ? (agreeCount / matched.length * 100).toFixed(1) + '%' : '-', lbl: 'AI一致率'},
  {num: (matched.length ? totalLoss / matched.length : 0).toFixed(1) + ' pt/判断', lbl: '平均相違度'},
  {num: majorCount + '件', lbl: '大相違', sub: `うち重要度高 ${majorHighStakesCount}件`},
  {num: worst ? `${worst.label} ${worst.junme}巡目` : '-', lbl: '最も相違した局面'},
];
document.getElementById('stat-tiles').innerHTML = tiles.map(t =>
  `<div class="stat-tile"><div class="num">${t.num}</div><div class="lbl">${t.lbl}</div>${t.sub ? `<div class="sub">${t.sub}</div>` : ''}</div>`).join('');

function chipHtml(tile, ring) {
  const cls = 'chip' + (ring ? ' ring' : '');
  const style = ring ? `border-color:${ring}` : '';
  return `<span class="${cls}" style="${style}">${tileLabel(tile)}</span>`;
}

function severityChip(sev) {
  const m = SEV_META[sev];
  return `<span class="severity-chip" style="border-color:${m.color}">${m.icon} ${m.label}</span>`;
}

// --- cumulative loss chart ---
(function() {
  const svg = document.getElementById('cum-chart');
  const W = svg.clientWidth || 900, H = 220, padL = 40, padR = 10, padT = 10, padB = 24;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const n = matched.length;
  if (n === 0) { document.getElementById('cum-caption').textContent = '判断データがありません。'; return; }
  let cum = 0;
  const pts = matched.map((d, i) => { cum += d.loss; return {x: i, y: cum, d}; });
  const maxY = Math.max(1, pts[pts.length - 1].y);
  const x = i => padL + (n <= 1 ? 0 : (i / (n - 1)) * (W - padL - padR));
  const y = v => H - padB - (v / maxY) * (H - padT - padB);

  let grid = '';
  for (let g = 0; g <= 4; g++) {
    const v = maxY * g / 4;
    grid += `<line class="grid-line" x1="${padL}" y1="${y(v)}" x2="${W-padR}" y2="${y(v)}"/>`;
    grid += `<text x="${padL - 6}" y="${y(v)+3}" text-anchor="end">${v.toFixed(0)}</text>`;
  }
  let boundaries = '';
  let prevKy = -1;
  matched.forEach((d, i) => {
    if (d.kyoku_index !== prevKy) {
      prevKy = d.kyoku_index;
      const xx = x(i);
      boundaries += `<line class="grid-line" x1="${xx}" y1="${padT}" x2="${xx}" y2="${H-padB}" stroke-dasharray="2,3"/>`;
      boundaries += `<text class="kyoku-label" x="${xx+2}" y="${padT+10}">${d.label}</text>`;
    }
  });
  const path = pts.map((p, i) => (i === 0 ? 'M' : 'L') + x(p.x).toFixed(1) + ',' + y(p.y).toFixed(1)).join(' ');
  svg.innerHTML = grid + boundaries +
    `<path class="series-line" d="${path}"/>` +
    `<line class="crosshair" id="cum-cross" y1="${padT}" y2="${H-padB}"/>`;

  // drop any kyoku-boundary label whose measured box would overlap the
  // previous kept label; the tick line above stays regardless.
  let lastRight = -Infinity;
  svg.querySelectorAll('.kyoku-label').forEach(t => {
    const bbox = t.getBBox();
    if (bbox.x < lastRight + 4) {
      t.remove();
    } else {
      lastRight = bbox.x + bbox.width;
    }
  });

  const tooltip = document.getElementById('cum-tooltip');
  const cross = document.getElementById('cum-cross');
  svg.addEventListener('mousemove', e => {
    const rect = svg.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width * W;
    let idx = Math.round((mx - padL) / (W - padL - padR) * (n - 1));
    idx = Math.max(0, Math.min(n - 1, idx));
    const p = pts[idx], d = p.d;
    const xx = x(idx);
    cross.style.display = 'block';
    cross.setAttribute('x1', xx);
    cross.setAttribute('x2', xx);
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX - rect.left + 12) + 'px';
    tooltip.style.top = (e.clientY - rect.top - 10) + 'px';
    tooltip.innerHTML = `${d.label} ${d.junme}巡目<br>あなた: ${tileLabel(d.mine_tile) || d.mine_label}<br>AI推奨: ${candText(d.candidates[0])}<br>ロス: ${d.loss.toFixed(1)}pt`;
  });
  svg.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; cross.style.display = 'none'; });

  const steepest = pts.reduce((a, b, i) => i > 0 && (b.y - pts[i-1].y) > a.jump ? {jump: b.y - pts[i-1].y, d: b.d} : a, {jump: 0, d: null});
  document.getElementById('cum-caption').textContent = steepest.d
    ? `AIと最も相違したのは ${steepest.d.label} ${steepest.d.junme}巡目（+${steepest.jump.toFixed(1)}pt）。急な立ち上がりはAIとの相違が続いた区間。`
    : '';
})();

// --- per-kyoku bar chart ---
(function() {
  const svg = document.getElementById('bar-chart');
  const byKyoku = {};
  matched.forEach(d => {
    if (!byKyoku[d.kyoku_index]) byKyoku[d.kyoku_index] = {label: d.label, loss: 0, worst: 'agree'};
    byKyoku[d.kyoku_index].loss += d.loss;
    const order = {agree: 0, minor: 1, notable: 2, major: 3};
    if (order[d.severity] > order[byKyoku[d.kyoku_index].worst]) byKyoku[d.kyoku_index].worst = d.severity;
  });
  const keys = Object.keys(byKyoku).sort((a, b) => a - b);
  const W = svg.clientWidth || 900, padL = 90, padR = 60, rowH = 26, gap = 2;
  const H = keys.length * (rowH + gap) + 10;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const maxLoss = Math.max(1, ...keys.map(k => byKyoku[k].loss));
  const barW = k => (byKyoku[k].loss / maxLoss) * (W - padL - padR);
  let out = '';
  keys.forEach((k, i) => {
    const b = byKyoku[k];
    const yy = i * (rowH + gap) + 5;
    out += `<text x="${padL - 8}" y="${yy + rowH/2 + 4}" text-anchor="end">${b.label}</text>`;
    out += `<rect data-i="${i}" x="${padL}" y="${yy}" width="${barW(k).toFixed(1)}" height="${rowH-4}" rx="4" class="bar-fill"/>`;
    out += `<text x="${padL + barW(k) + 6}" y="${yy + rowH/2 + 4}">${b.loss.toFixed(1)}</text>`;
  });
  svg.innerHTML = out;

  const tooltip = document.getElementById('bar-tooltip');
  svg.querySelectorAll('rect').forEach((rect, i) => {
    rect.addEventListener('mousemove', e => {
      const k = keys[i], b = byKyoku[k];
      const rectBB = svg.getBoundingClientRect();
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX - rectBB.left + 12) + 'px';
      tooltip.style.top = (e.clientY - rectBB.top - 10) + 'px';
      tooltip.innerHTML = `${b.label}<br>合計ロス: ${b.loss.toFixed(1)}pt<br>最悪の判断: ${SEV_META[b.worst].label}`;
    });
    rect.addEventListener('mouseleave', () => tooltip.style.display = 'none');
  });
})();

// --- decision cards ---
function renderCards(order) {
  const list = order === 'loss' ? [...matched].sort((a, b) => b.loss - a.loss) : [...matched].sort((a, b) => a.order - b.order);
  const html = list.map(d => {
    const collapsed = d.severity === 'agree';
    const topTile = d.candidates[0].tile;
    const handHtml = d.hand.map(t => {
      let ring = null;
      if (t === d.mine_tile) ring = SEV_META[d.severity].color;
      else if (t === topTile) ring = SEV_META.agree.color;
      return chipHtml(t, ring);
    }).join('');
    const candHtml = d.candidates.map(c => {
      const mineMatch = (d.mine_tile && c.tile === d.mine_tile) || (!d.mine_tile && c.label === d.mine_label);
      return `<div class="cand-line${mineMatch ? ' mine' : ''}">${mineMatch ? '→ ' : ''}${candDisplay(c)} — ${(c.prob*100).toFixed(1)}%</div>`;
    }).join('');
    const stakesBadge = `<span class="stakes-badge">重要度: ${d.stakes}</span>`;
    if (collapsed) {
      return `<div class="card collapsed">${severityChip(d.severity)}${stakesBadge}<span>${d.label} ${d.junme}巡目 シャンテン${d.shanten} — あなた: ${tileLabel(d.mine_tile) || d.mine_label}</span></div>`;
    }
    return `<div class="card">
      <div class="card-title">${severityChip(d.severity)} ${stakesBadge} ${d.label} ${d.junme}巡目 シャンテン${d.shanten}${d.has_meld ? ' ・鳴きあり' : ''} ・相違度 ${d.loss.toFixed(1)}pt</div>
      <div class="card-detail">
        <div>手牌: ${handHtml}</div>
        <div class="cand-line mine" style="margin-top:6px">あなた: ${d.mine_tile ? chipHtml(d.mine_tile, SEV_META[d.severity].color) : d.mine_label}</div>
        <div style="margin-top:4px">AI推奨:</div>
        ${candHtml}
      </div>
    </div>`;
  }).join('');
  document.getElementById('decision-list').innerHTML = html || '<p>判断データがありません。</p>';
}
let sortMode = 'loss';
renderCards(sortMode);
document.getElementById('sort-toggle').addEventListener('click', () => {
  sortMode = sortMode === 'loss' ? 'order' : 'loss';
  document.getElementById('sort-toggle').textContent = '並び順: ' + (sortMode === 'loss' ? '相違度順（切替）' : 'ゲーム順（切替）');
  renderCards(sortMode);
});

document.getElementById('footnote').innerHTML =
  `相違度は Mortal（自己学習モデル mortal_final.pth）が打牌候補に割り当てた方策確率のギャップ（percentage points）であり、AIが他の選択をどれだけ強く選好したかを示す指標です。証明された点数期待値（EV）の損失ではありません。` +
  `重大度（一致/小差/相違/大相違）の閾値、および重要度（高/中/低）の判定は、いずれもヒューリスティックな目安です。` +
  (DATA.unmatched > 0 ? `<br>${DATA.unmatched}件の判断は行動のマッピングに失敗したため集計から除外しました。` : '');
</script>
</body>
</html>
'''


def escape_for_script_tag(payload_json):
    return payload_json.replace('<', '\\u003c')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--seat', type=int, default=None)
    ap.add_argument('--out', default=None)
    here = os.path.dirname(os.path.abspath(__file__))
    fallback = '/Users/kosei/mahjang/Akagi/mjai_bot/mortal_kosei'
    default_bot_dir = here if os.path.exists(os.path.join(here, 'mortal_final.pth')) else fallback
    ap.add_argument('--bot-dir', default=default_bot_dir)
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    lines = read_lines(args.log)

    seat = args.seat
    if seat is None:
        for ln in lines:
            ev = json.loads(ln)
            if ev['type'] == 'start_game' and 'id' in ev:
                seat = ev['id']
                break
        if seat is None:
            sys.exit('start_game.id が見つかりません。--seat 0-3 を指定してください。')

    decisions, kyoku_count, unmatched = replay(lines, seat, args.bot_dir)

    title = args.title or os.path.basename(args.log)
    payload = {'title': title, 'seat': seat, 'kyoku_count': kyoku_count, 'decisions': decisions, 'unmatched': unmatched}
    payload_json = escape_for_script_tag(json.dumps(payload, ensure_ascii=False))

    if args.out:
        out_path = args.out
    else:
        base = args.log
        for suffix in ('.mjai.jsonl.gz', '.json.gz', '.mjai.jsonl', '.jsonl.gz', '.jsonl'):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        else:
            base = os.path.splitext(base)[0]
        out_path = base + '.html'
    out_path = os.path.abspath(out_path)

    html_out = HTML_TEMPLATE.replace('__TITLE__', html.escape(title)).replace('__PAYLOAD__', payload_json)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_out)

    print(out_path)


if __name__ == '__main__':
    main()
