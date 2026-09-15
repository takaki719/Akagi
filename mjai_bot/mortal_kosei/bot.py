import sys
import os
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import Brain, DQN
from engine import MortalEngine
from libriichi.mjai import Bot

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mortal_final.pth')

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


def build_engine():
    torch.set_num_threads(2)
    state = torch.load(MODEL_PATH, weights_only=True, map_location='cpu')
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


def resolve_seat():
    if len(sys.argv) > 1 and sys.argv[-1].lstrip('-').isdigit():
        return int(sys.argv[-1])
    return int(os.environ.get('AKAGI_PLAYER_ID', 0))


def add_show_card(reaction):
    meta = reaction.get('meta')
    if not meta or 'q_values' not in meta or 'mask_bits' not in meta:
        return reaction
    mask_bits = meta['mask_bits']
    q_values = meta['q_values']
    bits = [i for i in range(46) if mask_bits & (1 << i)]
    if len(bits) != len(q_values):
        print(f'meta mismatch: {len(bits)} set bits vs {len(q_values)} q_values', file=sys.stderr)
        return reaction
    qs = torch.tensor(q_values, dtype=torch.float64)
    probs = torch.softmax(qs, dim=0).tolist()
    ranked = sorted(zip(bits, probs), key=lambda x: -x[1])[:3]
    items = []
    for rank, (idx, p) in enumerate(ranked):
        if idx < 37:
            item = {'label': '打', 'pais': [DISCARD_TILES[idx]], 'value': f'{p * 100:.1f}%'}
        else:
            item = {'label': ACTION_LABELS[idx], 'value': f'{p * 100:.1f}%'}
        if rank == 0:
            item['color'] = '#00c853'
        items.append(item)
    shanten = meta.get('shanten')
    title = f'シャンテン {shanten}' if shanten is not None else '候補'
    meta['show'] = {'title': title, 'items': items}
    return reaction


def main():
    seat = resolve_seat()
    engine = build_engine()
    bot = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        saw_end_game = False
        try:
            events = json.loads(line)
            reaction = None
            for event in events:
                etype = event.get('type')
                if etype == 'start_game':
                    seat = event.get('id', seat)
                    bot = Bot(engine, seat)
                if etype == 'end_game':
                    saw_end_game = True
                if bot is None:
                    continue
                r = bot.react(json.dumps(event))
                if event is events[-1]:
                    reaction = r
            if reaction is None:
                out = {'type': 'none'}
            else:
                out = add_show_card(json.loads(reaction))
        except Exception:
            traceback.print_exc(file=sys.stderr)
            out = {'type': 'none'}

        sys.stdout.write(json.dumps(out, separators=(',', ':')) + '\n')
        sys.stdout.flush()

        if saw_end_game:
            break


if __name__ == '__main__':
    main()
