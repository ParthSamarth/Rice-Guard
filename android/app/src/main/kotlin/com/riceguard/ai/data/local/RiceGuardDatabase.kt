package com.riceguard.ai.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

/** Adds the source/notes/pipelineProcessingTimeSec/serverVersion columns
 * introduced for the Scan Index feature. All four are nullable (or have a
 * SQL default), so this is a plain additive migration -- existing scan rows
 * and their images are preserved, never wiped, on upgrade. */
private val MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE scan_records ADD COLUMN source TEXT NOT NULL DEFAULT 'camera'")
        db.execSQL("ALTER TABLE scan_records ADD COLUMN notes TEXT")
        db.execSQL("ALTER TABLE scan_records ADD COLUMN pipelineProcessingTimeSec REAL")
        db.execSQL("ALTER TABLE scan_records ADD COLUMN serverVersion TEXT")
    }
}

@Database(entities = [ScanRecord::class], version = 2, exportSchema = false)
abstract class RiceGuardDatabase : RoomDatabase() {
    abstract fun scanDao(): ScanDao

    companion object {
        @Volatile private var instance: RiceGuardDatabase? = null

        fun getInstance(context: Context): RiceGuardDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    RiceGuardDatabase::class.java,
                    "riceguard.db",
                )
                    .addMigrations(MIGRATION_1_2)
                    .build().also { instance = it }
            }
    }
}
