#include "mbed.h"
#include <cmath>
#include "DA7212.h"
#include "uLCD_4DGL.h"
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define bufferLength (32)

DA7212 audio;
uLCD_4DGL uLCD(D1, D0, D2);
DigitalOut greenLed(LED2);
InterruptIn pause(SW2);
DigitalIn button3(SW3);
int16_t waveform[kAudioTxBufferSize];
EventQueue queue(32 * EVENTS_EVENT_SIZE);
EventQueue display_queue(32 * EVENTS_EVENT_SIZE);
Thread t;
Thread DNN_thread(osPriorityNormal, 120*1024 /*120K stack size*/);
Thread display_thread;
Serial pc(USBTX, USBRX);

char serialInBuffer[bufferLength];
int serialCount = 0;
int idC = 0;
int mode = 0; 
char mode_name[5] = {"play", "forward", "backward", "change songs", "which song"};
char song_name[3] = {"star", "yamaha", "tiger"};
int which_song = 0;
int data_store = 0;
int store_i = 0;
int note;
int score = 0;
int signalLength[3] = {42, 34, 32};
int *song, *noteLength;

void playNote(int freq)
{
  for(int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}

void ulcd(void) {
  uLCD.cls();

  uLCD.locate(0,0);
  uLCD.printf("mode:%s",mode_name[mode]);

  uLCD.locate(0,1);
  uLCD.printf("%d : %s", which_song, song_name[which_song]);

  uLCD.locate(0,2);
  uLCD.printf("Takio beat: %d", note);

  uLCD.locate(0,3);
  uLCD.printf("Takio score: %d", score);
}

void modeSelection(void){
  if(mode == 0){
    mode = 1;
  }
  queue.cancel(idC);
  audio.spk.pause();
  display_queue.call(ulcd);
}

void loadSignal(void)
{
  song = signalLength[which_song];
  noteLength = song;
  
  pc.printf("%d\r\n", which_song);

  greenLed = 0;
  int i = 0;
  serialCount = 0;
  audio.spk.pause();
  while(i < song)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 7)
      {
        serialInBuffer[serialCount] = '\0';
        song[i] = (int) atof(serialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
  i = 0;
  while(i < song)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 5)
      {
        serialInBuffer[serialCount] = '\0';
        noteLength[i] = (int) atof(serialInBuffer);
        serialCount = 0;
        i++;
      }
    }
  }
  greenLed = 1;

  data_store = which_song;
}

void modeSelectionHandler(void) {queue.call(modeSelection);}

void loadSignalHandler(void) {queue.call(loadSignal);}

// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void DNN(){
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                             tflite::ops::micro::Register_RESHAPE(), 1);

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    return -1;
  }

  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      if(gesture_index == 0){
        if(mode == 4){
          which_song = 0;
        }
        else if(mode == 0){}
        else{
          mode = 1;
        }
      }
      else if(gesture_index == 1){
        if(mode == 4){
          which_song = 1;
        }
        else if(mode == 0){}
        else{
          mode=2;
        }
      }
      else if(gesture_index == 2){
        if (mode == 4){
          which_song = 2;
        }
        else if(mode == 0){
          if(note == 0)
            score += 1;
        }
        else{
          mode=3;
        }
      }
      else if(gesture_index == 3){
        if (mode == 4){
          if(note == 1){
            score += 1;
          }
        }
      }
      display_queue.call(ulcd);
    }
  }
}

int main(int argc, char* argv[])
{
  uLCD.background_color(0x000000);
  uLCD.color(WHITE);
  display_queue.call(ulcd);
  
  DNN_thread.start(DNN);

  greenLed = 1;
  loadSignal();
  t.start(callback(&queue, &EventQueue::dispatch_forever));
  display_thread.start(&display_queue, &EventQueue::dispatch_forever);

  pause.rise(queue.event(modeSelectionHandler));

  while (true) {
    if(mode == 0){
      if(which_song != data_store){
        loadSignal();
        store_i = 0;
        score = 0;
      }

      for(int i = store_i; i < signalLength[which_song]; i++)
      {
        store_i = i;
        int length = noteLength[i];

        if(length > 1){
          note = 0;
        }else{
          note = 1;
        }
        display_queue.call(ulcd);
        while(length--)
        {
          // the loop below will play the note for the duration of 1s
          for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize ; ++j)
          {
            idC = queue.call(playNote, song[i]);
          }
          if(length < 1) wait(1.0);
        }
      }
        
    }else{
      if(button3 == 0){
        if(mode == 1){
          if(which_song < 2){
            which_song += 1;
          }
          else if(which_song == 2){
            which_song = 0;
          }
          mode = 0;
        }
        else if(mode == 2){
          if(which_song > 0){
            which_song -= 1;
          }
          else if(which_song == 0){
            which_song = 2;
          }
          mode = 0;
        }
        else if(mode == 4){
          mode = 0;
        }
        else if(mode == 3){
          mode = 4;
        }
        display_queue.call(ulcd);
        wait(0.5);
      }
    }
  wait(0.1);
  }
}