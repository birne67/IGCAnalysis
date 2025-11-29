"""Quickstart examples for the IGCAnalysis package.

This file shows minimal, non-GUI usage so you can test the editable install
from your MyPythonLibrary copy quickly.
"""
from __future__ import annotations

import IGCAnalysis as pkg
from IGCAnalysis.myfunctions import haversine

def main() -> None:
    print('IGCAnalysis package file:', getattr(pkg, '__file__', '<none>'))
    print('IGCAnalysis version:', getattr(pkg, '__version__', '<no __version__>'))

    # quick haversine sample
    lon1, lat1 = 11.5, 49.4
    lon2, lat2 = 12.6, 48.8
    d_km = haversine(lon1, lat1, lon2, lat2)
    print(f'Distance example: {d_km:.2f} km (haversine)')

    # check for a couple of available symbols
    try:
        from IGCAnalysis import AsymPlotLib as apl
        print('AsymPlotLib loaded from', getattr(apl, '__file__', '<unknown>'))
        print('AsymPlotLib has readfile:', hasattr(apl, 'readfile'))
    except Exception as exc:  # import errors should not crash quickstart
        print('AsymPlotLib import failed:', exc)

if __name__ == '__main__':
    main()
