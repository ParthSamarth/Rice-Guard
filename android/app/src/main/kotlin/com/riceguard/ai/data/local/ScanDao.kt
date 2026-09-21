package com.riceguard.ai.data.local

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ScanDao {

    @Query("SELECT * FROM scan_records ORDER BY timestampMillis DESC")
    fun observeAll(): Flow<List<ScanRecord>>

    @Query("SELECT * FROM scan_records WHERE id = :id")
    suspend fun getById(id: String): ScanRecord?

    @Query("SELECT COUNT(*) FROM scan_records")
    fun observeCount(): Flow<Int>

    @Query("SELECT COUNT(*) FROM scan_records")
    suspend fun count(): Int

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(record: ScanRecord)

    @Query("UPDATE scan_records SET notes = :notes WHERE id = :id")
    suspend fun updateNotes(id: String, notes: String?)

    @Delete
    suspend fun delete(record: ScanRecord)

    @Query("DELETE FROM scan_records WHERE id = :id")
    suspend fun deleteById(id: String)

    @Query("DELETE FROM scan_records")
    suspend fun deleteAll()
}
