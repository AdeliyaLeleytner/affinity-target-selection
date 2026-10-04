"""Render deposited NR3C1/DEX coordinates; run beside 1M2Z.cif with PyMOL installed.

No coordinate generation, docking, geometry repair, contacts, or score overlay.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pymol2

BASE = Path(__file__).resolve().parent
SOURCE_HASH = 'ada431c0857e60ef8a1176a54add1ef29bab5c1bc87fda79d03157ab009e0692'
RECEPTOR = 'reference and chain A and polymer.protein and resi 523-777 and not hydro'
LIGAND = 'reference and chain A and resn DEX and resi 301 and not hydro'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    assert hashlib.sha256((BASE / '1M2Z.cif').read_bytes()).hexdigest() == SOURCE_HASH
    # Use the genuine pymol-open-source distribution (see environment.yml).
    # The embedded instance renders coordinates without an external GUI.
    with pymol2.PyMOL('presentation') as instance:
        cmd = instance.cmd
        cmd.set('max_threads', 2)
        cmd.set('auto_zoom', 0)
        cmd.viewport(1800, 1800)
        cmd.load(str(BASE / '1M2Z.cif'), 'reference')
        cmd.select('receptor', RECEPTOR)
        cmd.select('ligand', LIGAND)
        assert cmd.count_atoms('receptor') == 2074
        assert cmd.count_atoms('ligand') == 28
        cmd.hide('everything', 'all')
        cmd.set_color('receptor_blue', [0.40, 0.56, 0.69])
        cmd.set_color('ligand_orange', [0.90, 0.40, 0.12])
        cmd.color('receptor_blue', 'receptor')
        cmd.color('ligand_orange', 'ligand')
        cmd.show('cartoon', 'receptor')
        cmd.show('sticks', 'ligand')
        cmd.show('spheres', 'ligand')
        cmd.set('stick_radius', 0.22, 'ligand')
        cmd.set('sphere_scale', 0.23, 'ligand')
        cmd.set('cartoon_transparency', 0.14, 'receptor')
        cmd.set('cartoon_fancy_helices', 1)
        cmd.set('cartoon_smooth_loops', 1)
        cmd.set('cartoon_sampling', 14)
        cmd.set('cartoon_loop_radius', 0.20)
        cmd.set('orthoscopic', 1)
        cmd.set('depth_cue', 0)
        cmd.set('ray_shadows', 0)
        cmd.set('antialias', 2)
        cmd.set('ambient', .52)
        cmd.set('direct', .55)
        cmd.set('specular', .12)
        cmd.set('shininess', 20)
        cmd.set('two_sided_lighting', 1)
        cmd.set('ray_trace_mode', 0)
        cmd.bg_color('white')
        cmd.set('ray_opaque_background', 1)
        cmd.orient('receptor')
        cmd.turn('x', 15)
        cmd.turn('y', -20)
        cmd.turn('z', -15)
        cmd.zoom('receptor or ligand', buffer=4.0, complete=1)
        cmd.deselect()
        full_view = list(cmd.get_view())
        dimensions = (900, 900) if args.preview else (1800, 1800)
        name = 'nr3c1_dexamethasone_preview.png' if args.preview else 'nr3c1_dexamethasone_full.png'
        cmd.png(str(BASE / name), width=dimensions[0], height=dimensions[1], dpi=300, ray=1)
        if args.preview:
            return
        cmd.set('ray_opaque_background', 0)
        cmd.png(str(BASE / 'nr3c1_dexamethasone_full_transparent.png'), width=1800, height=1800, dpi=300, ray=1)
        # A closer framing uses the same deposited pose and camera orientation.
        # Only the framing changes; the complete receptor remains represented.
        cmd.set('ray_opaque_background', 1)
        cmd.viewport(1800, 1600)
        cmd.zoom('receptor or ligand', buffer=3.0, complete=1)
        pocket_view = list(cmd.get_view())
        cmd.png(str(BASE / 'nr3c1_dexamethasone_pocket.png'), width=1800, height=1600, dpi=300, ray=1)
        record = {'pymol_version': list(cmd.get_version()), 'receptor_selection': RECEPTOR,
                  'ligand_selection': LIGAND, 'receptor_atoms': cmd.count_atoms('receptor'),
                  'ligand_heavy_atoms': cmd.count_atoms('ligand'),
                  'full_camera': full_view, 'pocket_camera': pocket_view,
                  'backgrounds': ['white', 'transparent'], 'projection': 'orthographic',
                  'rendering': 'CPU ray tracing; two threads; raster molecular illustration',
                  'coordinates': 'Original deposited CIF; no structural editing or generated pose.'}
        (BASE / 'camera_settings.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
