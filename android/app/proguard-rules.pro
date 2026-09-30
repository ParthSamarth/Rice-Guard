# RiceGuard AI -- release is currently built with isMinifyEnabled = false
# (see app/build.gradle.kts), so these rules are not active yet. Kept ready
# for when a real release build is configured.

-keep class com.riceguard.ai.data.network.dto.** { *; }
-keep class com.riceguard.ai.data.local.** { *; }
