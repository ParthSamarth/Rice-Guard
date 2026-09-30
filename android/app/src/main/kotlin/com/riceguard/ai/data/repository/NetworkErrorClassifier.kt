package com.riceguard.ai.data.repository

import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException

/** Turns a network exception into one specific, actionable sentence, so a
 * failed request always tells the user WHY (wrong IP, wrong port, server not
 * running, or just a dropped Wi-Fi link) instead of one generic message for
 * every failure mode. Shared by [ConnectionRepository] and ScanRepository. */
fun describeConnectionFailure(e: Throwable): String = when (e) {
    is UnknownHostException ->
        "Invalid server address. Check the IP address in Settings."
    is ConnectException ->
        "Connection refused. Check that the server is running and the port is correct."
    is SocketTimeoutException ->
        "Connection timed out. Check the IP address and that your phone and PC are on the same Wi-Fi network."
    else ->
        "Could not reach the server. Check your Wi-Fi connection."
}
