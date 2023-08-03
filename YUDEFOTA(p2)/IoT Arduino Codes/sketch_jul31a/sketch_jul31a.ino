const int audioInPin = A0;
const int gainPin = 3;
const int sampleRate = 48000;  // Set the sample rate to 48 kHz
const int bufferSize = 1024;   // Set the buffer size to 1024 samples

void setup() {
  Serial.begin(115200);

  // Set gain to HIGH (60dB)
  pinMode(gainPin, OUTPUT);
  digitalWrite(gainPin, HIGH);
}

void loop() {
  // Buffer to store audio samples
  int audioBuffer[bufferSize];

  // Collect audio samples
  for (int i = 0; i < bufferSize; i++) {
    audioBuffer[i] = analogRead(audioInPin);
    delayMicroseconds(1000000 / sampleRate);  // Delay to achieve desired sample rate
  }

  // Send the audio samples to the Flask app via Serial
  for (int i = 0; i < bufferSize; i++) {
    Serial.print(audioBuffer[i]);
    if (i < bufferSize - 1) {
      Serial.print(",");
    }
  }
  Serial.println();

  // Optional: Add a small delay to avoid overwhelming the Serial communication
  delay(100);
}
