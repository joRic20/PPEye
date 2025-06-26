/* =======================================================================
 *  Nicla Vision PPE Detection - Serial Output Only (with Debug)
 * -----------------------------------------------------------------------
 *  • Captures 320 × 240 RGB565 frame
 *  • Resizes to model input size 
 *  • Runs PPE-detection model (Edge Impulse)
 *  • Shows results on Serial Monitor and LEDs
 *  • No LCD dependency - Serial output only
 * =======================================================================
 */

#include <vector>
#include <cstring>
#include <PPE_Detection_try3_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"
#include "camera.h"
#include "gc2145.h"

// Include ea_malloc.h only if available, otherwise use standard malloc
#ifdef ARDUINO_NICLA_VISION
#include <ea_malloc.h>
#define SAFE_MALLOC ea_malloc
#define SAFE_FREE ea_free
#else
#define SAFE_MALLOC malloc
#define SAFE_FREE free
#endif

/* ---------------- compile-time options ---------------- */
#define DETECTION_THRESHOLD  0.50f         // confidence threshold
#define CAM_W  320
#define CAM_H  240
#define SHOW_TIMINGS  1                    // 1 = print inference time

/* ---------------- helper types ----------------------- */
typedef struct { size_t w, h; } rez_t;

/* ---------------- LED definitions ------------------- */
#ifndef LEDR
#define LEDR LED_BUILTIN
#endif
#ifndef LEDG  
#define LEDG LED_BUILTIN
#endif

#define LED_RED  LEDR
#define LED_GRN  LEDG

/* ---------------- camera globals --------------------- */
GC2145 sensor;
Camera  cam(sensor);
FrameBuffer fb;

static uint8_t *raw565 = nullptr;                      // 320×240×2 = 154 kB
static uint8_t *rgb888 = nullptr;                      // model size ×3 (≈ 27 kB for 96×96)
static bool      cam_ok = false;

/* ---------------- forward declarations --------------- */
bool  camera_begin();
bool  capture_resize(uint32_t w, uint32_t h);
static int  image_cb(size_t off, size_t len, float *dst);
bool  rgb565_to_rgb888(uint8_t *src, uint8_t *dst, uint32_t len);
int   pick_resize(uint32_t ow,uint32_t oh,uint32_t *rw,uint32_t *rh,bool *dr);
static void fatalBlink();

/* ============================ SETUP ============================ */
void setup() {
    Serial.begin(115200);
    delay(2000); // Give serial time to initialize

    Serial.println("\n==== DEBUG: Entered setup() ====");
    
    Serial.println("========================================");
    Serial.println("    PPE DETECTION SYSTEM STARTING");
    Serial.println("========================================");
    
    // Initialize LED pins
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_GRN, OUTPUT);
    digitalWrite(LED_RED, LOW);
    digitalWrite(LED_GRN, LOW);
    
    Serial.println("✓ LEDs initialized");

#ifdef ARDUINO_NICLA_VISION
    /* give Nicla's M4 RAM to heap */
    malloc_addblock((void*)0x30000000, 288 * 1024);
    Serial.println("✓ Extended memory pool added");
#endif

    Serial.println("📷 Initializing camera...");
    if (!camera_begin()) { 
        Serial.println("❌ Camera initialization FAILED");
        fatalBlink(); 
    }
    Serial.println("✓ Camera initialized successfully");

    /* Allocate RGB888 buffer at model size with bounds checking */
    size_t rgb_len = (size_t)EI_CLASSIFIER_INPUT_WIDTH * 
                     (size_t)EI_CLASSIFIER_INPUT_HEIGHT * 3 + 32;
    
    Serial.print("🧠 Allocating RGB buffer: ");
    Serial.print(rgb_len);
    Serial.println(" bytes");
    
    uint8_t *temp_rgb888 = (uint8_t*)SAFE_MALLOC(rgb_len);
    if (!temp_rgb888) { 
        Serial.println("❌ RGB buffer allocation FAILED");
        fatalBlink(); 
    }
    
    // Align to 32-byte boundary
    rgb888 = (uint8_t*)(((uintptr_t)temp_rgb888 + 31) & ~31u);
    Serial.println("✓ RGB buffer allocated successfully");
    
    Serial.println("========================================");
    Serial.println("    SYSTEM READY - PPE DETECTION ON");
    Serial.println("========================================");
    Serial.print("Model input size: ");
    Serial.print(EI_CLASSIFIER_INPUT_WIDTH);
    Serial.print(" x ");
    Serial.println(EI_CLASSIFIER_INPUT_HEIGHT);
    Serial.print("Detection threshold: ");
    Serial.println(DETECTION_THRESHOLD);
    Serial.println();
}

/* ============================= LOOP ============================ */
void loop() {
    delay(3000); // 3 second delay between detections

    Serial.println("📸 Starting capture sequence...");
    Serial.print("Target size: ");
    Serial.print(EI_CLASSIFIER_INPUT_WIDTH);
    Serial.print(" x ");
    Serial.println(EI_CLASSIFIER_INPUT_HEIGHT);
    
    // Add heartbeat to show we're alive
    digitalWrite(LED_GRN, HIGH);
    delay(100);
    digitalWrite(LED_GRN, LOW);
    
    if (!capture_resize(EI_CLASSIFIER_INPUT_WIDTH,
                        EI_CLASSIFIER_INPUT_HEIGHT)) {
        Serial.println("❌ Frame capture failed - trying again...");
        digitalWrite(LED_RED, HIGH);
        delay(200);
        digitalWrite(LED_RED, LOW);
        return;
    }
    
    Serial.println("✓ Frame captured and resized");

    // Prepare Edge Impulse signal
    ei::signal_t sig;
    sig.total_length = (size_t)EI_CLASSIFIER_INPUT_WIDTH * (size_t)EI_CLASSIFIER_INPUT_HEIGHT;
    sig.get_data     = &image_cb;

    ei_impulse_result_t res = {};
    
#if SHOW_TIMINGS
    uint32_t t0 = millis();
#endif
    
    Serial.println("🧠 Running AI inference...");
    EI_IMPULSE_ERROR inference_result = run_classifier(&sig, &res, false);
    
    if (inference_result != EI_IMPULSE_OK) {
        Serial.print("❌ Inference error: ");
        Serial.println(inference_result);
        digitalWrite(LED_RED, HIGH);
        digitalWrite(LED_GRN, LOW);
        return;
    }
    
#if SHOW_TIMINGS
    uint32_t inference_time = millis() - t0;
    Serial.print("⏱️  Inference time: ");
    Serial.print(inference_time);
    Serial.println(" ms");
#endif

    // Analyze results
    std::vector<const char*> detected, missing;
    
    Serial.println("\n📊 DETECTION RESULTS:");
    Serial.println("----------------------------------------");
    
    // Check each class
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        float confidence = res.classification[i].value;
        const char* label = ei_classifier_inferencing_categories[i];
        
        Serial.print("  ");
        Serial.print(label);
        Serial.print(": ");
        Serial.print(confidence * 100, 1);
        Serial.print("%");
        
        if (confidence >= DETECTION_THRESHOLD) {
            detected.push_back(label);
            Serial.println(" ✓ DETECTED");
        } else {
            missing.push_back(label);
            Serial.println(" ✗ Missing");
        }
    }
    
    Serial.println("----------------------------------------");
    
    // Summary
    Serial.print("🎯 DETECTED PPE: ");
    if (detected.empty()) {
        Serial.println("None");
    } else {
        for (size_t i = 0; i < detected.size(); i++) {
            Serial.print(detected[i]);
            if (i + 1 < detected.size()) Serial.print(", ");
        }
        Serial.println();
    }
    
    Serial.print("⚠️  MISSING PPE: ");
    if (missing.empty()) {
        Serial.println("None");
    } else {
        for (size_t i = 0; i < missing.size(); i++) {
            Serial.print(missing[i]);
            if (i + 1 < missing.size()) Serial.print(", ");
        }
        Serial.println();
    }
    
    // Final verdict
    Serial.println("========================================");
    if (missing.empty()) {
        Serial.println("🟢 ACCESS GRANTED - All PPE detected!");
        digitalWrite(LED_GRN, HIGH); 
        digitalWrite(LED_RED, LOW);
    } else {
        Serial.println("🔴 ACCESS DENIED - Missing required PPE");
        digitalWrite(LED_RED, HIGH); 
        digitalWrite(LED_GRN, LOW);
    }
    Serial.println("========================================\n");
}

/* ===================== camera_begin() ========================== */
bool camera_begin() {
    if (cam_ok) return true;
    
    Serial.println("Starting camera initialization...");
    
    if (!cam.begin(CAMERA_R320x240, CAMERA_RGB565, -1)) {
        Serial.println("Camera begin() failed");
        return false;
    }

    delay(300); // sensor warm-up
    
    size_t raw_size = (size_t)CAM_W * (size_t)CAM_H * 2 + 32;
    uint8_t *temp_raw565 = (uint8_t*)SAFE_MALLOC(raw_size);
    if (!temp_raw565) {
        Serial.println("Raw buffer allocation failed");
        return false;
    }
    
    // Align to 32-byte boundary
    raw565 = (uint8_t*)(((uintptr_t)temp_raw565 + 31) & ~31u);
    
    fb.setBuffer(raw565);
    
    cam_ok = true;
    Serial.println("Camera initialization complete");
    return true;
}

/* ===================== capture_resize() ======================== */
bool capture_resize(uint32_t w, uint32_t h) {
    if (!cam_ok) {
        Serial.println("❌ Camera not initialized");
        return false;
    }
    
    Serial.println("🔄 Attempting frame grab...");
    int e = cam.grabFrame(fb, 3000); // 3 second timeout
    if (e != 0) { 
        Serial.print("❌ grabFrame error code: "); 
        Serial.println(e);
        Serial.println("Will retry next loop...");
        return false; 
    }
    Serial.println("✓ Frame grabbed successfully");

    size_t frame_size = cam.frameSize();
    Serial.print("📏 Frame size: ");
    Serial.print(frame_size);
    Serial.println(" bytes");
    
    if (frame_size == 0) {
        Serial.println("❌ Frame size is zero");
        return false;
    }

    Serial.println("🎨 Converting RGB565 to RGB888...");
    if (!rgb565_to_rgb888(raw565, rgb888, frame_size)) {
        Serial.println("❌ RGB conversion failed");
        return false;
    }
    Serial.println("✓ RGB conversion successful");

    uint32_t rw, rh; 
    bool dr;
    pick_resize(w, h, &rw, &rh, &dr);
    
    Serial.print("📐 Resize to: ");
    Serial.print(rw);
    Serial.print(" x ");
    Serial.print(rh);
    Serial.print(" (downscale: ");
    Serial.print(dr ? "yes" : "no");
    Serial.println(")");
    
    if (dr) {
        if (rw > 0 && rh > 0 && CAM_W > 0 && CAM_H > 0) {
            Serial.println("🔧 Cropping and interpolating...");
            ei::image::processing::crop_and_interpolate_rgb888(
                rgb888, CAM_W, CAM_H, rgb888, rw, rh);
            Serial.println("✓ Resize complete");
        } else {
            Serial.println("❌ Invalid resize parameters");
            return false;
        }
    }
    return true;
}

/* ============== Edge-Impulse image callback =================== */
static int image_cb(size_t off, size_t len, float *dst) {
    if (!rgb888 || !dst) return -1;
    
    size_t total_pixels = (size_t)EI_CLASSIFIER_INPUT_WIDTH * (size_t)EI_CLASSIFIER_INPUT_HEIGHT;
    
    if (off + len > total_pixels) {
        Serial.println("Image callback: bounds error");
        return -1;
    }
    
    size_t ix = off * 3;
    for (size_t i = 0; i < len; i++) {
        // Pack RGB into a single float as expected by Edge Impulse
        dst[i] = (float)((rgb888[ix] << 16) | (rgb888[ix + 1] << 8) | rgb888[ix + 2]);
        ix += 3;
    }
    return 0;
}

/* ================= RGB565 → RGB888 ============================ */
bool rgb565_to_rgb888(uint8_t *s, uint8_t *d, uint32_t len) {
    if (!s || !d || len == 0) return false;
    
    if (len % 2 != 0) {
        Serial.println("RGB565 conversion: odd length");
        return false;
    }
    
    uint32_t remaining = len;
    while (remaining >= 2) {
        uint8_t hi = *s++, lo = *s++;
        
        *d++ = (hi & 0xF8) | ((hi & 0xE0) >> 5);           // Red
        *d++ = ((hi & 0x07) << 5) | ((lo & 0xE0) >> 3);    // Green  
        *d++ = ((lo & 0x1F) << 3) | ((lo & 0x1C) >> 2);    // Blue
        
        remaining -= 2;
    }
    return true;
}

/* ================= pick_resize() ============================== */
int pick_resize(uint32_t ow, uint32_t oh, uint32_t *rw, uint32_t *rh, bool *dr) {
    if (!rw || !rh || !dr) return -1;
    
    static const rez_t tbl[] = {{64,64}, {96,96}, {160,120}, {160,160}, {320,240}};
    
    *rw = CAM_W; 
    *rh = CAM_H; 
    *dr = false;
    
    for (const auto &r : tbl) {
        if (ow <= r.w && oh <= r.h) { 
            *rw = r.w; 
            *rh = r.h; 
            *dr = true; 
            break; 
        }
    }
    return 0;
}

/* ================= fatalBlink() =============================== */
static void fatalBlink() {
    pinMode(LED_RED, OUTPUT);
    Serial.println("💀 FATAL ERROR - entering blink loop");
    
    while (true) { 
        digitalWrite(LED_RED, HIGH); 
        delay(200); 
        digitalWrite(LED_RED, LOW); 
        delay(200); 
    }
}
