# Portfolio Lab brand

The Portfolio Lab's mark: one original design, used as the Mac app's icon and,
in its simplified version, as the browser tab's icon, so the Dock and the tab
match. No trademarked marks, third-party brand assets or exchange logos are
used.

<img src="portfolio-lab-icon.svg" alt="Portfolio Lab icon" width="160">

## The idea

A price line rises and falls up to a dashed vertical "now" line, the decision
time: nothing to its right is known yet. From there a forecast fan opens, an
outer and an inner probability band around a dashed median. The half-width of
each band grows with the square root of the horizon (each edge is a quadratic
curve), the way forecast uncertainty usually grows. The mark carries two of the
Lab's habits: nothing after "now" is drawn as known, and a forecast comes with
its uncertainty.

## Files

| File | Use |
|---|---|
| `portfolio-lab-icon.svg` | The app icon from 64 pixels up: full detail, full bleed |
| `portfolio-lab-icon-small.svg` | The app icon at 16 and 32 pixels: no grid, median or "now" line; a thicker, three-segment line |
| `portfolio-lab-mark.svg` | The small artwork on a rounded square, for places that do not round it themselves |
| `src/stock_agent/app/static/portfolio-lab-icon.png` | The browser-tab icon (64 by 64 pixels), rendered from the mark |

The two icon SVGs are full bleed on purpose. On macOS 27 (tested on a Retina
display), the system masks a full-bleed icon into its own rounded square. Our
first draft had transparent margins and a drop shadow, and macOS placed it on a
grey tile instead.

The PNG iconset (16 to 1024 pixels) and the `.icns` are not stored here. The
Mac app's build script (`scripts/build_mac_app.py`) renders them from these SVGs
with macOS's `sips` and `iconutil` each time it builds the app, outside the
repository.

## Colours

| Role | Colour | Source |
|---|---|---|
| Surface (gradient) | `#352D26` to `#13110F` | brand only (near the dark canvas) |
| Grid, "now" line | `#F3EEE6` | the dark scheme's text colour |
| Price line, median | `#F6F1E9`, `#FFF6EA` | brand only (ivory) |
| Last price | `#FFFDF8` | the light scheme's surface colour |
| Outer band, glow | `#DE8C66` | the dark scheme's accent (terracotta) |
| Inner band | `#F2BC72` | the dark scheme's amber |

The theme tokens are in `src/stock_agent/ui/theme.py`;
`tests/test_brand_assets.py` keeps the SVGs to this list. Amber is the dark
scheme's warning colour, so it stays in the logo and is not used to decorate
the interface.

Two variations were considered and not used: Terracotta (a `#C2622F` to
`#77301B` tile with ivory line and bands) and Ivory (an `#FFFDF8` tile with a
charcoal line and `#9A4325` bands). Either is a colour-only change to these
files.

## Changing the artwork

Edit the SVGs (keep the 1024 by 1024 canvas), then render the browser-tab icon
again from the repository root:

```bash
sips -s format png -z 64 64 assets/brand/portfolio-lab-mark.svg --out src/stock_agent/app/static/portfolio-lab-icon.png
```

Rebuild the Mac app to update its icon. Keep the small artwork legible at 16
pixels: check it at actual size before committing.

Safari keeps its own cache of each site's tab icon. If Safari still shows
Streamlit's default icon for the Lab's address, clear that cache (or open the
Lab in another browser once) rather than changing the app.
