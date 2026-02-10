# GitHub Pages Deployment Guide

This folder contains the complete project homepage. You can upload it directly to GitHub and enable GitHub Pages.

## Deployment steps

### Method 0 (Recommended): GitHub Actions → GitHub Pages
1. Ensure these files exist in the repo root: `index.html`, `styles.css`, `main.js` (they do in this repo).
2. Push to `main`.
3. Go to Settings → Pages.
4. Under “Build and deployment”, set **Source** to **GitHub Actions**.
5. Wait for the workflow “Deploy static site to GitHub Pages” to finish (Actions tab).

### Method 1: Upload via GitHub UI
1. Create a new repository (or use an existing one).
2. Upload the contents of `homepage` to the repo.
3. Go to Settings → Pages.
4. Set Source to “Deploy from a branch”.
5. Pick the branch (usually `main` or `master`).
6. Select folder `/ (root)`.
7. Save.

### Method 2: Use Git CLI
```bash
cd homepage
git init
git add .
git commit -m "Initial commit: Add homepage"
git branch -M main
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```
Then enable GitHub Pages in repo settings.

## Notes
1. **Video file size**: GitHub file limit is ~100MB. If larger, consider:
   - Git LFS (Large File Storage)
   - External storage/CDN (e.g., Cloudflare, AWS S3)
   - Compression

2. **Git LFS setup** (if needed):
   ```bash
   git lfs install
   git lfs track "*.mp4"
   git add .gitattributes
   ```

3. **Custom domain** (optional):
   - Add a `CNAME` file in `homepage`
   - Add your domain, e.g., `yourdomain.com`

4. **HTTPS**: Provided automatically by GitHub Pages.

## Accessing the site

After deployment, your homepage will be available at:
- `https://your-username.github.io/your-repo/`

If the repo name matches your username:
- `https://your-username.github.io/`

## Updating content

After changes:
```bash
git add .
git commit -m "Update homepage"
git push
```

GitHub Pages will redeploy automatically (usually a few minutes).
