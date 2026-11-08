#ifdef LOG

SdFat     Card;
SdFile    FlightLog_csv;
unsigned long id = 1;

// ---------------------------------------------------------------------------
// Logging policy (stage-A audit fix P0).
//
// The legacy implementation did, on *every* 100 Hz control tick:
//     open("flightlog.csv", O_RDWR|O_CREAT|O_AT_END) -> build a String -> println ->
//     close()
// Three problems, all of which are fatal for a 100 Hz loop:
//   1. a single open/write/close on an SD card can take tens of milliseconds, so the
//      control period was being overrun by the logger;
//   2. a dozen String heap allocations per sample, i.e. heap fragmentation over a
//      long flight;
//   3. with O_AT_END every power-up appended another header line, so the CSV ended up
//      with several headers and downstream parsers broke.
//
// New policy: the file is opened once and kept open; the header is written only when
// the file is empty (first power-up on a fresh card); rows are written straight into
// the file's buffer and the data is flushed with sync() every LOG_FLUSH_PERIOD
// samples (0.5 s at 100 Hz), which bounds what a power loss can cost without
// stalling the loop.
//
// The CSV columns are unchanged (downstream parsers depend on them).
// ---------------------------------------------------------------------------
#define LOG_FLUSH_PERIOD      50
static unsigned int log_since_flush = 0;
static bool         log_ready = false;

void Log_init() {
  pinMode(CSPin, OUTPUT);
  log_ready = false;
  if (Card.begin(CSPin, SPI_HALF_SPEED)) {
    if (FlightLog_csv.open("flightlog.csv", O_RDWR | O_CREAT | O_AT_END)) {
      log_ready = true;
      if (FlightLog_csv.fileSize() == 0) {  // fresh file -> write the header once
#ifdef STR_v1
        FlightLog_csv.println(F("ID, RollD, PitchD, YawD, Roll, Pitch, Yaw, RateD, Rate, a1, a2, b0, b1"));
#else
        FlightLog_csv.println(F("ID, RollD, PitchD, YawD, Roll, Pitch, Yaw"));
#endif
        FlightLog_csv.sync();
      }
    }
  }
}

// Flush accumulated rows to the card.  Called periodically from Log_savedata().
void Log_flush() {
  if (!log_ready) return;
  FlightLog_csv.sync();
}

void Log_savedata() {
  if (!log_ready) return;
  if (!FlightLog_csv.isOpen()) {
    // The card was removed or the file handle was lost: try once to reopen, but do
    // not let logging block the control loop.
    log_ready = FlightLog_csv.open("flightlog.csv", O_RDWR | O_CREAT | O_AT_END);
    if (!log_ready) return;
  }
  char line[192];
  int n;
#ifdef STR_v1
  n = snprintf(line, sizeof(line),
               "%lu, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.4f, %.4f, %.4f, %.4f",
               id,
               (double)(Angle_eulerdesire[IndexRoll]),
               (double)(Angle_eulerdesire[IndexPitch]),
               (double)(Angle_eulerdesire[IndexYaw]),
               (double)(Angle_newestimate[IndexRoll] * RadToDeg),
               (double)(Angle_newestimate[IndexPitch] * RadToDeg),
               (double)(Angle_newestimate[IndexYaw] * RadToDeg),
               (double)(Rate_middledesire[STR_Index]),
               (double)(Rate_threemeasure[STR_Index] * RadToDeg),
               (double)(a1), (double)(a2), (double)(b0), (double)(b1));
#else
  n = snprintf(line, sizeof(line), "%lu, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f",
               id,
               (double)(Angle_eulerdesire[IndexRoll]),
               (double)(Angle_eulerdesire[IndexPitch]),
               (double)(Angle_eulerdesire[IndexYaw]),
               (double)(Angle_newestimate[IndexRoll] * RadToDeg),
               (double)(Angle_newestimate[IndexPitch] * RadToDeg),
               (double)(Angle_newestimate[IndexYaw] * RadToDeg));
#endif
  if (n <= 0) return;
  FlightLog_csv.println(line);
  id++;
  if (++log_since_flush >= LOG_FLUSH_PERIOD) {
    log_since_flush = 0;
    Log_flush();
  }
}

#endif
