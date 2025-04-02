# Hospital Management System

A modern web-based hospital management system built with Node.js, Express, and EJS templates.

## Features

- User authentication (Login/Signup)
- Role-based access (Patient, Doctor, Admin)
- Book appointments
- View bookings
- Admin dashboard
- Modern UI with responsive design

## Requirements

- Node.js (v14+)
- MySQL or SQLite (choose one)

## Installation

1. Clone the repository
   ```
   git clone https://github.com/PREETHAM1590/HMS.git
   ```
2. Navigate to the project directory
3. Install dependencies:
   ```
   npm install
   ```

## Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Database configuration
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=hospital1

# Session configuration
SESSION_SECRET=hospitalmanagement2025secretkey

# Server configuration
PORT=3000
```

## Database Setup

### Option 1: MySQL

1. Make sure MySQL server is running
2. Create a database (it will be created automatically when you run the server if it doesn't exist)
3. Run the server:
   ```
   node server.js
   ```

### Option 2: SQLite (No MySQL required)

If you're having issues with MySQL or prefer not to use it, you can use the SQLite version:

1. Install SQLite dependencies:
   ```
   npm install sqlite3 sqlite --save
   ```
2. Run the SQLite version of the server:
   ```
   node server-sqlite.js
   ```

The SQLite database will be created automatically in the `data` directory.

## Running the Application

### Using MySQL
```
node server.js
```

### Using SQLite
```
node server-sqlite.js
```

The server will start on port 3000 (or the port specified in your .env file). Visit http://localhost:3000 in your browser.

## Deployment to Vercel

This project is configured for deployment on Vercel:

1. Fork or clone this repository to your GitHub account
2. Create a new project on [Vercel](https://vercel.com)
3. Import your GitHub repository
4. Configure environment variables in the Vercel dashboard:
   - `DATABASE_URL`: Your production database URL (MySQL or PlanetScale)
   - `SESSION_SECRET`: A secure random string for session encryption
   - `NODE_ENV`: Set to `production`
5. Deploy!

### Using MySQL with Vercel

For production deployment, you'll need a MySQL-compatible database that's accessible from Vercel:
- [PlanetScale](https://planetscale.com/)
- [Amazon RDS](https://aws.amazon.com/rds/mysql/)
- [Digital Ocean Managed MySQL](https://www.digitalocean.com/products/managed-databases)

## Default Admin Login (SQLite version only)

Email: admin@hospital.com
Password: admin123

## Troubleshooting Database Issues

### Testing MySQL Connection

Run the test script to check if your MySQL connection is working:
```
node test-db.js
```

### Viewing SQLite Database Contents

Run the following script to view the contents of the SQLite database:
```
node view-sqlite-db.js
```

## Project Structure

- `server.js` - Main server file (MySQL version)
- `server-sqlite.js` - Alternative server file using SQLite
- `db-sqlite.js` - SQLite database adapter
- `templates/` - EJS templates for rendering pages
  - `components/` - Reusable UI components
  - `*.ejs` - Page templates
- `static/` - Static assets
  - `images/` - Image files
  - `js/` - JavaScript files
- `public/` - Public assets
  - `css/` - CSS files
- `data/` - SQLite database files (created when using SQLite)
- `vercel.json` - Configuration for Vercel deployment

## Credits

Developed as a modern replacement for legacy hospital management systems. 