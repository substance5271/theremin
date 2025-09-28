import cv2
import numpy as np
import sounddevice as sd
import threading

# Camera
cap = cv2.VideoCapture(0)
prev_frame = None

# Sound parameters
fs = 44100  # Sampling rate
stream = None
current_freq = 440
current_vol = 0.0

# Thread-safe buffer for sound generation
lock = threading.Lock()
freq_vol = [current_freq, current_vol]

def audio_callback(outdata, frames, time, status):
    with lock:
        f, v = freq_vol
    t = (np.arange(frames) + audio_callback.pos) / fs
    outdata[:] = (np.sin(2 * np.pi * f * t) * v).reshape(-1,1)
    audio_callback.pos += frames
audio_callback.pos = 0

# Start audio stream
stream = sd.OutputStream(channels=1, callback=audio_callback, samplerate=fs)
stream.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Motion detection
    if prev_frame is not None:
        delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # Map x to frequency (200Hz to 1000Hz)
            freq = np.interp(x + w//2, [0, frame.shape[1]], [200, 1000])
            # Map y to volume (0 to 1)
            vol = 1 - np.interp(y + h//2, [0, frame.shape[0]], [0, 1])

            with lock:
                freq_vol[0] = freq
                freq_vol[1] = vol

            # Draw rectangle over detected motion
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    prev_frame = gray.copy()

    # Display info
    cv2.putText(frame, f"Freq: {int(freq_vol[0])} Hz", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Vol: {freq_vol[1]:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Theremin", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# Cleanup
stream.stop()
stream.close()
cap.release()
cv2.destroyAllWindows()

