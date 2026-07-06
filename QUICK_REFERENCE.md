# 📚 Quick Reference

This document provides quick access to the most commonly used commands and information for your portfolio application.

## 🚀 Quick Start Commands

### Local Development

```bash
# Start the application locally
python app.py

# Generate a new SECRET_KEY
python generate_secret_key.py
```

### Production Deployment

```bash
# Deploy to Vercel
python migrate_to_postgres.py
git add .
git commit -m "Migrate to Vercel Postgres"
git push
```

### Data Migration

```powershell
# Windows PowerShell
$env:POSTGRES_URL="your-postgres-url-from-vercel"
python migrate_to_postgres.py
```

```bash
# Unix/Linux/macOS
export POSTGRES_URL="your-postgres-url-from-vercel"
python migrate_to_postgres.py
```

## 📁 File Management

### Upload Files

**Local Development:**
- Files are saved to `static/uploads/` directory
- Automatically creates unique filenames with UUID

**Production:**
- Files are uploaded to Vercel Blob storage
- Returns CDN URLs for file access

### File Structure

```
protfolio/
├── app.py                    # Main application
├── db_config.py              # Database abstraction
├── storage_helper.py         # File storage abstraction
├── migrate_to_postgres.py    # Migration script
├── generate_secret_key.py    # SECRET_KEY generator
├── rag_module.py             # RAG module (optional)
├── data.db                   # SQLite database (development)
├── static/                   # Static files
│   └── uploads/              # Uploaded files
├── templates/                # HTML templates
├── requirements.txt          # Python dependencies
├── README.md                 # Quick start guide
├── DEPLOYMENT_GUIDE.md       # Complete deployment guide
├── MIGRATION_SUMMARY.md      # Migration overview
├── QUICK_REFERENCE.md        # This file
└── CODE_CHANGES.md           # Detailed code changes
```

## 🔧 Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `POSTGRES_URL` | PostgreSQL connection string | `postgres://user:pass@host:5432/db` |
| `BLOB_READ_WRITE_TOKEN` | Vercel Blob storage token | `token_1234567890abcdef` |
| `SECRET_KEY` | Flask secret key | `your-secret-key-here` |
| `ADMIN_PASSWORD` | Admin password | `admin123` |
| `FLASK_ENV` | Flask environment | `production` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google AI API key | None |

### Setting Environment Variables

**Vercel (Production):**
1. Go to https://vercel.com/dashboard
2. Select your project → **Settings** → **Environment Variables**
3. Add each variable with its value

**Local Development (.env file):**
```bash
POSTGRES_URL=your-postgres-url
BLOB_READ_WRITE_TOKEN=your-blob-token
SECRET_KEY=your-secret-key
ADMIN_PASSWORD=your-admin-password
FLASK_ENV=development
```

## 🛠️ Development Commands

### Database Operations

**Initialize SQLite Database (Development):**
```bash
# The database is initialized automatically when app.py starts
python app.py
```

**Migrate to PostgreSQL (Production):**
```bash
python migrate_to_postgres.py
```

### File Operations

**Upload a file via admin panel:**
1. Login to `/admin`
2. Navigate to the appropriate section (Profile, Projects, etc.)
3. Use the file upload controls
4. Files are automatically uploaded to the appropriate storage

**Delete a file:**
- Files are automatically deleted when their parent record is deleted
- Manual cleanup may be needed for orphaned files

## 🌐 Application Routes

### Public Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page (portfolio display) |
| `/contact` | POST | Submit contact form |

### Admin Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/admin/login` | GET/POST | Admin login page |
| `/admin/logout` | GET | Admin logout |
| `/admin` | GET | Admin dashboard |
| `/admin/profile` | POST | Update profile |
| `/admin/projects/add` | POST | Add new project |
| `/admin/projects/edit/<id>` | GET/POST | Edit project |
| `/admin/projects/delete/<id>` | POST | Delete project |
| `/admin/experience/add` | POST | Add experience |
| `/admin/experience/edit/<id>` | GET/POST | Edit experience |
| `/admin/experience/delete/<id>` | POST | Delete experience |
| `/admin/skills/add` | POST | Add skill |
| `/admin/skills/edit/<id>` | GET/POST | Edit skill |
| `/admin/skills/delete/<id>` | POST | Delete skill |
| `/admin/knowledge` | POST | Upload knowledge PDF |
| `/admin/research/add` | POST | Add research paper |
| `/admin/research/edit/<id>` | GET/POST | Edit research paper |
| `/admin/research/delete/<id>` | POST | Delete research paper |
| `/admin/messages/delete/<id>` | POST | Delete message |

### API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/api/chat` | POST | Chat with knowledge base |

## 📊 Database Tables

### Core Tables

| Table | Purpose |
|-------|---------|
| `profile` | User profile information |
| `projects` | Portfolio projects |
| `experience` | Work experience entries |
| `skills` | Technical skills |
| `settings` | Application settings |

### Supporting Tables

| Table | Purpose |
|-------|---------|
| `messages` | Contact form submissions |
| `research_papers` | Research publications |

## 🔍 Database Operations

### Common Queries

**Get profile information:**
```sql
SELECT * FROM profile WHERE id = 1;
```

**Get all projects (newest first):**
```sql
SELECT * FROM projects ORDER BY datetime(created_at) DESC;
```

**Get skills (exploring first):**
```sql
SELECT * FROM skills ORDER BY exploring ASC, name ASC;
```

**Get recent messages:**
```sql
SELECT * FROM messages ORDER BY datetime(timestamp) DESC;
```

## 📝 File Upload Guidelines

### Supported File Types

| Type | Extensions | Size Limit |
|------|------------|------------|
| Images | `.jpg`, `.jpeg`, `.png`, `.gif`, `.svg` | 10MB |
| PDFs | `.pdf` | 50MB |
| Documents | `.doc`, `.docx`, `.txt` | 10MB |

### File Naming

- Files are automatically renamed with UUID prefix to prevent conflicts
- Original filename is preserved in metadata
- Secure filename handling prevents path traversal attacks

## 🐛 Common Issues and Solutions

### "Internal Server Error" when saving

**Problem:** Files or database operations fail in production

**Solution:**
1. Ensure environment variables are set correctly
2. Verify PostgreSQL and Blob storage are configured
3. Check Vercel deployment logs
4. Run migration script if needed

### Session keeps logging out

**Problem:** User sessions expire on deployment

**Solution:**
1. Ensure `SECRET_KEY` is set in environment variables
2. Use permanent SECRET_KEY across deployments
3. Check session configuration

### Can't upload files

**Problem:** File uploads fail

**Solution:**
1. Ensure `BLOB_READ_WRITE_TOKEN` is set
2. Verify file size and type limits
3. Check Vercel Blob storage configuration

### Migration script fails

**Problem:** Data migration encounters errors

**Solution:**
1. Ensure `POSTGRES_URL` is correct
2. Verify PostgreSQL database exists
3. Check database permissions
4. Run with proper error handling

## 📈 Performance Tips

### Database Optimization

- Use PostgreSQL in production for better performance
- Index frequently queried columns
- Use connection pooling for high-traffic scenarios

### File Storage Optimization

- Use Vercel Blob for production (scalable)
- Compress images before upload
- Set appropriate cache headers

### Application Optimization

- Enable Flask caching for static assets
- Use gzip compression
- Monitor database query performance

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

## 🔧 Technical Commands

### Development Environment

```bash
# Check Python version
python --version

# Install dependencies
pip install -r requirements.txt

# Check environment variables
env | grep -E "(POSTGRES|BLOB|SECRET|ADMIN|FLASK)"

# Test database connection
python -c "from db_config import get_db; print('Database connection successful')"
```

### Production Environment

```bash
# Check environment variables
vercel env ls

# View deployment logs
vercel logs

# Check database status
# Depends on your PostgreSQL provider
```

## 📚 Additional Resources

### Documentation

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Complete deployment instructions
- [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) - Migration overview
- [README.md](README.md) - Quick start guide

### Vercel Documentation

- [PostgreSQL Documentation](https://vercel.com/docs/storage/postgresql)
- [Blob Documentation](https://vercel.com/docs/storage/blob)
- [Environment Variables](https://vercel.com/docs/environment-variables)

### Flask Documentation

- [Flask Security](https://flask.palletsprojects.com/en/stable/security/)
- [Flask Sessions](https://flask.palletsprojects.com/en/stable/sessions/)

## 🎯 Quick Checklist

### Before Deployment

- [ ] Create Vercel Postgres database
- [ ] Create Vercel Blob storage
- [ ] Generate SECRET_KEY
- [ ] Set all environment variables
- [ ] Test local development
- [ ] Run migration script
- [ ] Verify data migration
- [ ] Deploy to Vercel
- [ ] Test production functionality

### After Deployment

- [ ] Verify file uploads work
- [ ] Test admin login and functionality
- [ ] Check contact form submissions
- [ ] Verify chat functionality
- [ ] Monitor performance metrics
- [ ] Set up monitoring and alerts

## 🔄 Version Information

### Current Version

- **Database:** PostgreSQL (production) / SQLite (development)
- **File Storage:** Vercel Blob (production) / Local (development)
- **Framework:** Flask
- **Architecture:** Modular with environment detection

### Version History

1.0 - Initial SQLite-only implementation
1.1 - Added PostgreSQL and Blob Storage support
1.2 - Added migration script and documentation
1.3 - Enhanced error handling and security

## 📞 Support

If you encounter issues:

1. Check Vercel deployment logs
2. Review environment variable settings
3. Verify database and storage configuration
4. Check file permissions
5. Refer to the detailed documentation

For additional help, contact support or refer to the documentation.

---

**Your portfolio is now production-ready!** 🚀