#ifndef IMU_H
#define IMU_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Fusion/Fusion.h>

FusionEuler getEuler();

typedef struct RawIMU {
    float ax, ay, az;
    float gx, gy, gz;
} RawIMU;

typedef struct MotionState {
    float ax, ay, az, vx, vy, vz, px, py, pz;
    float qw, qx, qy, qz;
} MotionState;

typedef struct IMU {
    RawIMU raw;
    MotionState state;
} IMU;


extern float features[1800];
extern bool features_ready;

#ifdef __cplusplus
}
#endif

#endif // IMU_H