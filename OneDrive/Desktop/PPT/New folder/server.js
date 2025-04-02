// Database setup - Connect without database name first
const dbConfig = {
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
};

// In production, use connection string if available
const dbName = process.env.DB_NAME || 'hospital1';
const isProd = process.env.NODE_ENV === 'production';

// Check and setup database
const setupDatabase = async () => {
    // Skip database creation in production environments
    if (isProd && process.env.DATABASE_URL) {
        console.log('Using production database connection');
        return;
    }
    
    try {
        // First connect without database name to create it if needed
        const rootPool = mysql.createPool(dbConfig);
        
        // ... existing code ...
    } catch (error) {
        console.error('Database setup error:', error);
        console.error('Please make sure MySQL is running and check your credentials');
    }
};

// Main database connection pool (with database name)
const pool = isProd && process.env.DATABASE_URL 
    ? mysql.createPool(process.env.DATABASE_URL)
    : mysql.createPool({
        ...dbConfig,
        database: dbName
    }); 