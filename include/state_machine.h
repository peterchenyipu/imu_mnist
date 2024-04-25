#ifndef STATE_MACHINE_H
#define STATE_MACHINE_H

#ifdef __cplusplus
extern "C" {
#endif

enum State {
    START,
    PAIR,
    PAIR_DISPLAY,
    DISCONNECT_DISPLAY,
    IDLE,
    COLLECT_DATA,
    INFERENCE,
    DISPLAY_INFERENCE
};

extern State state;



#ifdef __cplusplus
}
#endif

#endif // STATE_MACHINE_H