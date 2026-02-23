#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <EGL/egl.h>
#include <GL/gl.h>
#include <X11/Xlib.h>

#include <NvInfer.h>

using namespace nvinfer1;

// ============================================================
// Globals
// ============================================================

static int W = 640;
static int H = 480;
static int PIXELS = 0;

static unsigned char* d_bgr[2];
static float* d_trt[2];
static void* d_out[2];

static cudaStream_t stream;
static cudaEvent_t ready[2];
static int cur = 0;

static Display* x_display = nullptr;
static Window win;
static EGLDisplay eglDisplay = EGL_NO_DISPLAY;
static EGLContext eglContext = EGL_NO_CONTEXT;
static EGLSurface eglSurface = EGL_NO_SURFACE;

static GLuint tex = 0;
static cudaGraphicsResource* cuda_tex = nullptr;

static IRuntime* runtime = nullptr;
static ICudaEngine* engine = nullptr;
static IExecutionContext* context = nullptr;

static const char* inputTensorName = "input_0";
static const char* outputTensorName = "output_0";

// ============================================================
// Logger
// ============================================================

class Logger : public ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING)
            printf("[TRT] %s\n", msg);
    }
} gLogger;

#define CUDA_CHECK(x) \
    if ((x) != cudaSuccess) { \
        printf("CUDA error %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(x)); \
        return -1; \
    }

// ============================================================
// Helpers
// ============================================================

static size_t volumeDims(const Dims& d)
{
    size_t v = 1;
    for (int i = 0; i < d.nbDims; i++)
        v *= d.d[i];
    return v;
}

static size_t elementSize(DataType t)
{
    switch (t)
    {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF:  return 2;
        case DataType::kINT8:  return 1;
        case DataType::kINT32: return 4;
        case DataType::kBOOL:  return 1;
        default: return 4;
    }
}

// ============================================================
// CUDA Kernel: BGR -> RGBA8 (GL) + NCHW float (TRT)
// ============================================================
__global__ void bgr_to_surface_and_nchw(
    const unsigned char* __restrict__ bgr,
    cudaSurfaceObject_t surf,
    float* __restrict__ nchw,
    int W, int H, int pixels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    int i = idx * 3;

    unsigned char b = bgr[i + 0];
    unsigned char g = bgr[i + 1];
    unsigned char r = bgr[i + 2];

    // Optional debug (single pixel only)
    // if (x==0 && y==0)
    //     printf("BGR = %d %d %d\n", b,g,r);

    // Write RGBA8 (OpenGL)
    //uchar4 out = make_uchar4(r, g, b, 255);
    //uchar4 out = make_uchar4(255, 0, 0, 255); // DEBUG
    //surf2Dwrite(out, surf, x * sizeof(uchar4), (H - 1 - y));

    // TensorRT NCHW normalized
    float s = 1.0f / 255.0f;
    nchw[idx]            = r * s;
    nchw[pixels + idx]   = g * s;
    nchw[2*pixels + idx] = b * s;
}

__global__ void nchw_to_surface(
    const float* __restrict__ nchw,
    cudaSurfaceObject_t surf,
    int W, int H, int pixels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    
    // Read NCHW float channels and denormalize (0.0-1.0 -> 0-255)
    // Note: This matches the 'Sigmoid' output of your PyTorch Net
    unsigned char r = (unsigned char)(nchw[idx]            * 255.0f);
    unsigned char g = (unsigned char)(nchw[pixels + idx]   * 255.0f);
    unsigned char b = (unsigned char)(nchw[2*pixels + idx] * 255.0f);

    // Write back to the OpenGL surface (using the same flip logic as your BGR kernel)
    uchar4 out = make_uchar4(r, g, b, 255);
    surf2Dwrite(out, surf, x * sizeof(uchar4), (H - 1 - y));
}


// ============================================================
// TensorRT
// ============================================================

static int load_engine(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);

    void* data = malloc(size);
    fread(data, 1, size, f);
    fclose(f);

    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(data, size);
    free(data);

    if (!engine) return -1;

    context = engine->createExecutionContext();
    printf("TensorRT engine loaded\n");
    return 0;
}

static size_t setup_bindings()
{
    int nb = engine->getNbIOTensors();
    size_t output_bytes = 0;

    for (int i = 0; i < nb; i++)
    {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);

        if (mode == TensorIOMode::kINPUT)
        {
            Dims dims = engine->getTensorShape(name);
            bool dynamic = false;
            for (int d = 0; d < dims.nbDims; d++)
                if (dims.d[d] == -1) dynamic = true;

            if (dynamic)
                context->setInputShape(name, Dims4{1,3,H,W});
        }
    }

    for (int i = 0; i < nb; i++)
    {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto dtype = engine->getTensorDataType(name);

        if (mode == TensorIOMode::kOUTPUT)
        {
            Dims dims = context->getTensorShape(name);
            output_bytes = volumeDims(dims) * elementSize(dtype);
        }
    }

    return output_bytes;
}

// ============================================================
// EGL / GL
// ============================================================

static int init_x11_egl()
{
    x_display = XOpenDisplay(NULL);
    if (!x_display) return -1;

    Window root = DefaultRootWindow(x_display);

    XSetWindowAttributes swa;
    swa.event_mask = ExposureMask;
    win = XCreateWindow(x_display, root, 0,0,W,H,0,
                        CopyFromParent, InputOutput,
                        CopyFromParent, CWEventMask, &swa);

    XMapWindow(x_display, win);
    XFlush(x_display);

    eglDisplay = eglGetDisplay((EGLNativeDisplayType)x_display);
    eglInitialize(eglDisplay, NULL, NULL);

    const EGLint attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE,   8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE,  8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 0,
        EGL_NONE
    };

    EGLConfig config;
    EGLint num;
    eglChooseConfig(eglDisplay, attribs, &config, 1, &num);
    if (num == 0) {
        printf("eglChooseConfig failed\n");
        return -1;
    }

    if (!eglBindAPI(EGL_OPENGL_API)) {
        printf("eglBindAPI failed\n");
        return -1;
    }

    eglContext = eglCreateContext(eglDisplay, config, EGL_NO_CONTEXT, NULL);
    eglSurface = eglCreateWindowSurface(eglDisplay, config,
                                         (EGLNativeWindowType)win, NULL);

    eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);
    return 0;
}

static int init_gl()
{
    glViewport(0,0,W,H);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1,1,-1,1,-1,1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) printf("GL error: 0x%x\n", err);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glClearColor(1.0f, 0.0f, 1.0f, 1.0f); // magenta background test

    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cuda_tex, tex, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard));

    return 0;
}

// ============================================================
// Public API
// ============================================================

extern "C" {

int server_init(const char* engine_path, int width, int height)
{
    W = width;
    H = height;
    PIXELS = W * H;

    // ------------------------------------------------------------
    // 1. Create X11 + EGL context
    // ------------------------------------------------------------
    if (init_x11_egl() != 0)
    {
        printf("init_x11_egl failed\n");
        return -1;
    }

    // Must make EGL current BEFORE CUDA-GL interop
    if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext))
    {
        printf("eglMakeCurrent failed in server_init\n");
        return -1;
    }

    printf("GL_VENDOR   : %s\n", glGetString(GL_VENDOR));
    printf("GL_RENDERER : %s\n", glGetString(GL_RENDERER));
    printf("GL_VERSION  : %s\n", glGetString(GL_VERSION));

    // ------------------------------------------------------------
    // 2. Select CUDA device that matches the GL context
    // ------------------------------------------------------------
    unsigned int count = 0;
    int cudaDevice = 0;

    cudaError_t err = cudaGLGetDevices(
        &count,
        &cudaDevice,
        1,
        cudaGLDeviceListCurrentFrame
    );

    if (err != cudaSuccess || count == 0)
    {
        printf("cudaGLGetDevices failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("CUDA using GL device %d\n", cudaDevice);

    CUDA_CHECK(cudaSetDevice(cudaDevice));

    // ------------------------------------------------------------
    // 3. Initialize OpenGL objects (register texture AFTER device set)
    // ------------------------------------------------------------
    if (init_gl() != 0)
    {
        printf("init_gl failed\n");
        return -1;
    }

    // ------------------------------------------------------------
    // 4. Load TensorRT engine
    // ------------------------------------------------------------
    if (load_engine(engine_path) != 0)
    {
        printf("TensorRT load failed\n");
        return -1;
    }

    size_t out_bytes = setup_bindings();

    // ------------------------------------------------------------
    // 5. Create CUDA resources
    // ------------------------------------------------------------
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < 2; i++)
    {
        CUDA_CHECK(cudaMalloc(&d_bgr[i], PIXELS * 3));
        CUDA_CHECK(cudaMalloc(&d_trt[i], PIXELS * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out[i], out_bytes));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming));
    }

    printf("server_init complete (%dx%d)\n", W, H);
    return 0;
}

// ============================================================
// server_process_frame with test pattern
// ============================================================
int server_process_frame(unsigned char* host_frame, int size)
{
    if (!host_frame || size != PIXELS * 3) return -1;

    static FILE* train_file = nullptr;
    static float* host_float = nullptr;
    const int CHANNELS = 3;

    int next = 1 - cur;

    if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) return -1;

    // 1. Copy to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_bgr[next], host_frame, size, cudaMemcpyHostToDevice, stream));

    // 2. Map Surface and Run Kernel
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_tex, stream));
    cudaArray_t array;
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, cuda_tex, 0, 0));

    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = array;
    cudaSurfaceObject_t surf = 0;
    CUDA_CHECK(cudaCreateSurfaceObject(&surf, &desc));

    dim3 block(16,16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    bgr_to_surface_and_nchw<<<grid, block, 0, stream>>>(d_bgr[next], surf, d_trt[next], W, H, PIXELS);



    // --- 1. TensorRT Inference ---
    // Link the pre-processed d_trt[next] to the engine's input node
    context->setTensorAddress(inputTensorName, d_trt[next]);
    // Link the output buffer d_out[next] to the engine's output node
    context->setTensorAddress(outputTensorName, d_out[next]);

    // Run inference asynchronously on your existing stream
    if (!context->enqueueV3(stream)) {
        fprintf(stderr, "TensorRT inference failed!\n");
    }

    // --- 2. Reverse Kernel: float NCHW -> RGBA Surface ---
    // Launch a new kernel to take the AI output and write it back to the OpenGL surface
    // (Assuming d_out[next] is a float* of size PIXELS * 3)
    nchw_to_surface<<<grid, block, 0, stream>>>(
        (float*)d_out[next], surf, W, H, PIXELS);

    //nchw_to_surface<<<grid, block, 0, stream>>>( //DEBUG
    //    (float*)d_trt[next], surf, W, H, PIXELS); //DEBUG




    // 3. Cleanup CUDA Surface
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaDestroySurfaceObject(surf);
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_tex, stream));

    // 4. --- RENDER SECTION ---
    // Clear and set viewport every frame to prevent state drift
    glViewport(0, 0, W, H);
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f); // Dark gray background
    glClear(GL_COLOR_BUFFER_BIT);

    // Ensure we are in 2D coordinate mode
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    // Corrected Texture coordinates:
    // (0,0) is bottom-left in GL. Your kernel writes (H-1-y), 
    // so we map 0,0 to -1,-1 directly.
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    // 5. Swap
    if (!eglSwapBuffers(eglDisplay, eglSurface))
        printf("eglSwapBuffers failed!\n");


    
    // ----------------------------
    // Training export
    // ----------------------------
    if (!train_file) {
        train_file = fopen("/dev/shm/nn_frames.bin", "wb");
        if (!train_file) {
            fprintf(stderr, "Failed to open training file\n");
        }
    }

    if (train_file) {
        if (!host_float) {
            host_float = (float*)malloc(PIXELS * CHANNELS * sizeof(float));
            if (!host_float) {
                fprintf(stderr, "Failed to allocate host_float\n");
            }
        }

        if (host_float) {
            // Copy from NCHW float buffer (d_trt[next])
            CUDA_CHECK(cudaMemcpyAsync(
                host_float,
                d_trt[next],
                PIXELS * CHANNELS * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream));

            CUDA_CHECK(cudaStreamSynchronize(stream));

            fwrite(host_float, sizeof(float), PIXELS * CHANNELS, train_file);
            fflush(train_file);
        }
    }
    
    
    cur = next;
    return 0;
}

void server_cleanup()
{
    if (cuda_tex) cudaGraphicsUnregisterResource(cuda_tex);
    if (tex) glDeleteTextures(1,&tex);

    for (int i=0;i<2;i++)
    {
        if (d_bgr[i]) cudaFree(d_bgr[i]);
        if (d_trt[i]) cudaFree(d_trt[i]);
        if (d_out[i]) cudaFree(d_out[i]);
        cudaEventDestroy(ready[i]);
    }

    cudaStreamDestroy(stream);

    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;

    eglMakeCurrent(eglDisplay,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
    eglDestroySurface(eglDisplay,eglSurface);
    eglDestroyContext(eglDisplay,eglContext);
    eglTerminate(eglDisplay);

    XDestroyWindow(x_display,win);
    XCloseDisplay(x_display);
}

} // extern "C"