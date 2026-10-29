"""Build a public split preview from recorded dataset images; no simulation rerun."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


def build(dataset, output):
    output.mkdir(parents=True, exist_ok=False)
    index = json.loads((dataset / 'index.json').read_text())
    splits = ('train', 'validation', 'test')
    selected = []
    for split in splits:
        entries = [e for e in index['episodes'] if e['split'] == split]
        groups = list(dict.fromkeys(e['physical_group'] for e in entries))
        for row, group in enumerate(groups[:2]):
            variants = [e for e in entries if e['physical_group'] == group]
            selected.append((splits.index(split), row, variants[-1 if row else 0]))
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    width, height, fps = 1152, 508, 8
    duration = max(e['frames'] / 20 for _, _, e in selected)
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '21', '-pix_fmt',
               'yuv420p', '-movflags', '+faststart', str(output / 'splits.mp4')]
    sources = hashlib.sha256()
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for tick in range(round(duration * fps)):
            canvas = Image.new('RGB', (width, height), '#151b24')
            draw = ImageDraw.Draw(canvas)
            for column, split in enumerate(splits):
                count = sum(e['split'] == split for e in index['episodes'])
                title = f'{split.upper()} | {count} episodes'
                draw.text((column * 384 + 12, 10), title, font=font, fill='white')
                draw.text((column * 384 + 12, 35), 'Front camera          Inspection camera', font=small, fill='#c4cedb')
            for column, row, entry in selected:
                frame = min(round(tick / fps * 20), entry['frames'] - 1)
                x, y = column * 384, 60 + row * 224
                for camera_index, camera in enumerate(('front', 'inspection')):
                    relative = Path(entry['episode']) / 'images' / f'observation.images.{camera}' / f'{frame:06d}.jpg'
                    data = (dataset / relative).read_bytes()
                    sources.update(str(relative).encode() + data)
                    with Image.open(dataset / relative) as picture:
                        canvas.paste(picture.resize((192, 192), Image.Resampling.LANCZOS), (x + camera_index * 192, y))
                label = f"Group {entry['physical_group']:02d} / {Path(entry['episode']).name} | {frame / 20:.2f} s"
                draw.text((x + 9, y + 198), label, font=small, fill='white')
            if tick == round(6 * fps):
                canvas.save(output / 'splits.jpg', quality=92)
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError('Video encoding failed')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(output / 'splits.mp4'),
                    '-vf', 'fps=6,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer',
                    '-loop', '0', str(output / 'splits.gif')], check=True)
    manifest = dict(dataset_manifest_sha256=hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest(),
                    selection='First two physical groups in each split; first visual variant of first group and last variant of second group.',
                    episodes=[e for _, _, e in selected], fps=fps,
                    timing='Recorded at 20 Hz; preview samples at 8 fps in real time. Shorter clips hold their final frame.',
                    image_stream_sha256=sources.hexdigest(),
                    note='Recorded reference-data images, not trained-policy rollouts. No new simulation or evaluation.')
    (output / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.dataset, args.output)
