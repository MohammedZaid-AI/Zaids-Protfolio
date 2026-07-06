## Overview
Implement requested homepage changes in `templates/index.html` and adjust styles in `static/styles.css`.

## Changes to Apply
- Remove the Admin link from the hero socials on the homepage.
  - Delete `<a class="admin-link" href="{{ url_for('admin_login') }}">Admin</a>` at templates/index.html:35.
- Update the hero lead text.
  - Replace `Passionate developer with a dream` at templates/index.html:29 with `building products that dont suck`.
- Add a badge image in the hero section (below the lead line).
  - Insert a small badge block right under the lead paragraph:
    - `<div class="hero-badge"><img src="tos-alisg-i-84wi3idyod-sg/sg/7535424992205489159/image/1763186033085_cpq3ww0zkd40_png_297x64" alt="badge"></div>`.
  - Note: The provided path isn’t a full URL. If it doesn’t render, we can switch it to a valid `https://...` URL or place the image under `static/uploads` and reference it via `{{ url_for('static', filename='uploads/<file>') }}`.

## Style Updates
- Add minimal CSS for the new badge:
  - `.hero-badge { margin-top: 12px; }`
  - `.hero-badge img { width: 297px; height: auto; display: block; }`

## Admin Access
- The admin page remains reachable at `/admin/login`; it just won’t be linked from the homepage.

## Verification
- Start the app and load `http://127.0.0.1:5000/`.
- Confirm the Admin link is gone, the lead text reads "building products that dont suck", and the badge displays beneath it.

## After Confirmation
- I will apply the edits, run the app locally, and share the preview URL for you to review the result.