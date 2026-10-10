# people interface

The public page and decision workspace use a warm, restrained visual system for government and enterprise simulation. English is the default interface language; the EN / 中文 switch persists across both pages. Scenario names, user briefs, and engine or model output retain their original language.

## Visual direction

Warm porcelain (`#f6f3ee`), marine navy (`#203652`), and champagne copper (`#805b37`) define the interface. The theme stays light across sections and operating-system settings. DM Sans is self-hosted, with system Chinese font fallbacks. Rounded actions, eight-pixel controls, and softly outlined panels follow a consistent geometry.

The landing page moves from a decision-focused headline and architectural city model to the rehearsal process, government / business use cases, and a workspace entry. The workspace retains the existing simulation controls and their IDs, while adding section navigation, explicit mode selection, readable chart labels, and expandable diagnostics. Initial onboarding dialogs open only when requested.

## Review and refinements

| Dimension | Implemented and checked |
| --- | --- |
| Typography | Two-line hero, restrained display scale, readable English copy, corrected spacing around inline emphasis, and translated interface labels. |
| Whitespace | Consistent section gutters and panel spacing; advanced configuration stays collapsed until needed. |
| Visual hierarchy | Scenario setup and live results lead the workspace; raw diagnostics are expandable below the charts. |
| Color | A single navy / copper system across navigation, controls, charts, and network views; contrast refined with automated WCAG AA checks. |
| Motion | Scroll reveals introduce content; hover and state transitions provide feedback. Reduced-motion settings disable decorative movement and chart / network camera animation. |
| Micro-interactions | Keyboard-accessible use-case tabs, modal focus management, explicit mode controls, timeline event buttons for touch / keyboard use, and removable event chips alongside drag-and-drop. |
| Responsive layout | Landing and workspace checked at 320, 390, 768, 1024, 1440, and 1920 pixels, with no document-level horizontal overflow. The mobile section navigation scrolls within its own container. |
| Originality | A purpose-made city model connects civic institutions, businesses, communities, and branching infrastructure. No fabricated client logos, testimonials, or performance claims. |

The award references are a design-quality target, not a claim of an award or an independent jury assessment.

## Validation

- All 174 original element IDs and all 61 original input / select / textarea IDs are retained, in their original field order.
- Playwright checks cover English / Chinese switching, stored preferences, preserved user input, tabs, professional / guided modes, dialogs, focus restoration, and keyboard / touch timeline actions.
- axe-core 4.10.3 reports no WCAG 2 A / AA or WCAG 2.1 A / AA violations in the checked landing, workspace, guided mode, post-run, and guide-dialog states at desktop and mobile widths.
- Three-round Dry-Run simulations complete through the real SSE endpoint, with working indicators, narrative, and actor network. The interface language switch updates chart labels without changing narrative content. Paid model calls were not part of this verification.
- The 44 Python tests pass. Two existing simulation defects found while validating the coupled backend were fixed: initial regulatory investigations were appended to a discarded empty list, and ROI calculation read a KPI from the wrong state location. The KPI regression fixture now explicitly enables its extension and supplies a customer cohort.

The current workspace uses locally developed checkpoint, runtime-event, schema, evaluation, and audit capabilities. Their required people modules and associated regression tests accompany this interface update so a fresh repository checkout can run the same page.

## Assets and hosting

`web/assets/decision-city.webp` is an original image generated for this project using the built-in image generation tool, then resized to 1200 × 800 and encoded as WebP. The direction was: an elevated studio photograph of a physical architectural city model, with civic, business, and residential districts, warm limestone, navy smoked glass, and champagne-colored branching paths; no text or labels. It is conceptual imagery, not a depiction of a real project or simulation output.

DM Sans, Chart.js 4.4.1, and vis-network 9.1.9 are self-hosted with their license notices. Initial rendering does not need a font or script CDN. Source assets live in the repository rather than an external generation cache.

Start the app with:

```sh
python -m venv .venv
# Activate the virtual environment using your shell's activation command.
python -m pip install -r requirements.txt
python -m uvicorn people.webapp:app --host 127.0.0.1 --port 8770
```

The public page is `/`, and the workspace is `/people/`. FastAPI serves the shared assets at `/people/assets/` and `/assets/`, supporting gateways that retain or strip the `/people` prefix. A gateway serving the HTML independently must route those asset URLs to the same backend. Updating GitHub source does not itself restart an existing deployment.
