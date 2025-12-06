# Setup Guide

## Step 1: Installation (2 minutes)

```bash
pip install -r requirements.txt
```

## Step 2: Add PDFs (1 minute)

Place your 7 research papers in `pdfs/` folder.

## Step 3: Get API Key (30 seconds)

1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

## Step 4: Run (10 seconds)

```bash
streamlit run app.py
```

## Step 5: Configure (1 minute)

In the app:
1. Paste API key
2. Enter PDF folder path
3. Click "Load & Process Papers"

## Done! 🎉

Start asking questions about your papers!

## Troubleshooting

**Issue:** Module not found
**Fix:** `pip install -r requirements.txt`

**Issue:** No PDFs found
**Fix:** Check folder path, use full absolute path

**Issue:** API error
**Fix:** Verify API key is correct
