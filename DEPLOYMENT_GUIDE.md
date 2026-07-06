# 📚 Deployment Guide

This guide provides complete instructions for deploying your portfolio to production using Vercel with PostgreSQL and Blob Storage.

## 🚀 Quick Start

### Step 1: Create Vercel Postgres Database

1. Go to https://vercel.com/dashboard
2. Select your project → **Storage** tab
3. **Create Database** → **Postgres**
4. Name: "portfolio-db" → Create

### Step 2: Create Vercel Blob Storage

1. Same **Storage** tab
2. **Create Database** → **Blob**
3. Name: "portfolio-uploads" → Create

### Step 3: Set Environment Variables

Go to **Settings** → **Environment Variables**

Generate SECRET_KEY:
```bash
python generate_secret_key.py
```

Add these variables:
- `SECRET_KEY` = (generated key)
- `ADMIN_PASSWORD` = (your admin password)
- `GOOGLE_API_KEY` = (if using AI chat)
- `FLASK_ENV` = production

### Step 4: Migrate Your Data

```powershell
# Get POSTGRES_URL from Vercel Storage → Postgres → Settings
$env:POSTGRES_URL="your-postgres-url-from-vercel"

# Run migration
python migrate_to_postgres.py
```

### Step 5: Deploy

```bash
git add .
git commit -m "Migrate to Vercel Postgres"
git push
```

**Done!** 🎉

## 🧪 Test Your Deployment

1. Visit your Vercel URL
2. Go to `/admin/login`
3. Login with your ADMIN_PASSWORD
4. Try to save something (profile, project, etc.)
5. **It should work without errors!** ✅

## 🔧 Advanced Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `POSTGRES_URL` | Yes | PostgreSQL connection string from Vercel |
| `BLOB_READ_WRITE_TOKEN` | Yes | Vercel Blob storage token |
| `SECRET_KEY` | Yes | Flask secret key for sessions |
| `ADMIN_PASSWORD` | Yes | Admin password for login |
| `GOOGLE_API_KEY` | Optional | Google AI API key for chat |
| `FLASK_ENV` | Yes | Set to "production" |

### PostgreSQL Database Schema

The database contains the following tables:

- `profile` - User profile information
- `projects` - Portfolio projects
- `experience` - Work experience entries
- `skills` - Technical skills
- `settings` - Application settings
- `messages` - Contact form submissions
- `research_papers` - Research publications

### Blob Storage

All file uploads (avatars, project images, PDFs) are stored in Vercel Blob storage. This ensures:

- ✅ Persistent storage across deployments
- ✅ Scalable file handling
- ✅ CDN delivery
- ✅ No local filesystem dependencies

## 🐛 Troubleshooting

### "Internal Server Error" when saving

**Check:**
1. ✅ Postgres database created?
2. ✅ Blob storage created?
3. ✅ Environment variables set?
4. ✅ Code deployed?

**View logs:**
Vercel Dashboard → Deployments → [Latest] → View Function Logs

### Session keeps logging out

**Fix:** Make sure `SECRET_KEY` is set in Vercel environment variables

### Can't upload files

**Fix:** Make sure Blob storage is created and `BLOB_READ_WRITE_TOKEN` is set

### Migration script fails

**Fix:**
1. Ensure `POSTGRES_URL` is set correctly
2. Check that the PostgreSQL database exists
3. Run the migration script with proper permissions

## 📊 Monitoring

### Vercel Dashboard

- **Analytics**: View traffic and performance metrics
- **Functions**: Monitor API endpoint performance
- **Storage**: Check database and blob usage
- **Logs**: Debug application issues

### Local Development

For local development, the app automatically uses SQLite and local storage:

```bash
python app.py
```

No additional configuration needed for local development!

## 🔄 Rollback

If you need to rollback to SQLite:

1. Remove environment variables:
   - `POSTGRES_URL`
   - `BLOB_READ_WRITE_TOKEN`

2. The app will automatically fall back to SQLite
3. All data remains in the local `data.db` file

## 📈 Performance Tips

### Database Optimization

- Use PostgreSQL in production for better performance
- Enable connection pooling if needed
- Regularly clean up old messages and logs

### File Storage Optimization

- Use Vercel Blob for production (scalable)
- Compress images before upload
- Set appropriate cache headers

### Caching

- Enable Vercel caching for static assets
- Consider Redis for session storage in high-traffic scenarios

## 🛡️ Security Best Practices

### Environment Variables

- Never commit environment variables to git
- Use Vercel's environment variable management
- Rotate SECRET_KEY periodically

### File Uploads

- Validate file types and sizes
- Use secure_filename for file names
- Store files in secure storage (Vercel Blob)

### Database Access

- Use parameterized queries to prevent SQL injection
- Limit database permissions
- Regularly backup your database

## 📚 Additional Resources

- [Vercel PostgreSQL Documentation](https://vercel.com/docs/storage/postgresql)
- [Vercel Blob Documentation](https://vercel.com/docs/storage/blob)
- [Flask Security Best Practices](https://flask.palletsprojects.com/en/stable/security/)

## 🔧 Technical Details

### Architecture

The app uses a modular architecture with:

- **Database Abstraction**: `db_config.py` handles SQLite ↔ PostgreSQL
- **Storage Abstraction**: `storage_helper.py` handles local ↔ Blob
- **Lazy Loading**: `rag_module.py` loads only when needed

### Environment Detection

The app automatically detects its environment:

**Local Development:**
- Uses SQLite database
- Saves files to `static/uploads/`
- No configuration needed!

**Production (Vercel):**
- Uses PostgreSQL database
- Saves files to Blob Storage
- Auto-detected via environment variables

**Same code, works everywhere!**

## 📞 Support

If you encounter issues:

1. Check Vercel deployment logs
2. Review environment variable settings
3. Verify database and storage configuration
4. Check file permissions

For additional help, refer to the other documentation files in this repository.

---

**Your portfolio is now production-ready!** 🚀