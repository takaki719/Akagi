import gzip
import json
import os
import subprocess

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE = '/Users/kosei/mahjang/sample_game.json.gz'


def main():
    with gzip.open(SAMPLE, 'rt') as f:
        events = [json.loads(l) for l in f if l.strip()]

    proc = subprocess.Popen(
        ['.venv/bin/python', 'bot.py', '0'],
        cwd=BOT_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    batch = []
    sent = 0
    dahai_count = 0
    show_count = 0
    tsumo0_dahai_like = 0

    def send(batch):
        nonlocal sent
        proc.stdin.write(json.dumps(batch) + '\n')
        proc.stdin.flush()
        reply_line = proc.stdout.readline()
        assert reply_line, f'no reply for batch ending in {batch[-1]}'
        sent += 1
        return json.loads(reply_line)

    for event in events:
        batch.append(event)
        is_tsumo0 = event.get('type') == 'tsumo' and event.get('actor') == 0
        is_trigger = is_tsumo0 or event.get('type') in ('start_game', 'end_game')
        if is_trigger:
            reply = send(batch)
            assert isinstance(reply, dict) and 'type' in reply
            batch = []
            if is_tsumo0:
                if reply.get('type') in ('dahai', 'reach', 'hora'):
                    tsumo0_dahai_like += 1
                show = reply.get('meta', {}).get('show') if isinstance(reply.get('meta'), dict) else None
                if show and 1 <= len(show.get('items', [])) <= 3:
                    if all(item.get('value', '').endswith('%') for item in show['items']):
                        show_count += 1
            if reply.get('type') == 'dahai':
                dahai_count += 1
            if event.get('type') == 'end_game':
                break

    if batch:
        send(batch)

    proc.stdin.close()
    proc.wait(timeout=10)
    assert proc.stdout.read() == '', 'bot emitted extra stdout lines'

    print(f'replies={sent} dahai={dahai_count} tsumo0_actionable={tsumo0_dahai_like} show_cards={show_count}')
    assert tsumo0_dahai_like >= 5, f'expected >=5 actionable replies to actor-0 tsumo, got {tsumo0_dahai_like}'
    assert show_count >= 1, 'expected at least one reply with a valid show card'
    print('PASS')


if __name__ == '__main__':
    main()
