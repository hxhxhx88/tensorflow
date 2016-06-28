// Use following command to build, since for now I can not figure out how to build with bazel.
//  clang++ -std=c++11 -shared tensorflow/core/user_ops/resize_bilinear_3d_op.cc -o ~/Desktop/resize_bilinear_3d_op.so -fPIC -I $TF_INC -I . -undefined dynamic_lookup
// where 
//  TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

#define EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/image_resizer_state_3d.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

REGISTER_OP("ResizeBilinear3D")
    .Input("images: T")
    .Input("size: int32")
    .Output("resized_images: float")
    .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
    .Attr("align_corners: bool = false");

REGISTER_OP("ResizeBilinear3DGrad")
    .Input("grads_wrt_output: T")
    .Input("input_size: int32")
    .Output("grads_wrt_input: T")
    .Attr("T: {float, half, double}")
    .Attr("align_corners: bool = false");

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class ResizeBilinear3DOp : public OpKernel {
public:
    explicit ResizeBilinear3DOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    }

    void Compute(OpKernelContext* context) override {
        ImageResizerState3D st(align_corners_);
        st.ValidateAndCreateOutput(context);
        if (!context->status().ok()) return;

        typename TTypes<T, 5>::ConstTensor input_data = context->input(0).tensor<T, 5>();
        typename TTypes<float, 5>::Tensor output_data = st.output->tensor<float, 5>();

        for (int b = 0; b < st.batch_size; ++b) {
            for (int z = 0; z < st.out_depth; ++z) {
                const float in_z = z * st.depth_scale;
                const int64 front_z_index = static_cast<int64>(floorf(in_z));
                const int64 back_z_index = std::min(static_cast<int64>(ceilf(in_z)), (st.in_depth - 1));
                const float z_lerp = in_z - front_z_index;
                for (int y = 0; y < st.out_height; ++y) {
                    const float in_y = y * st.height_scale;
                    const int64 top_y_index = static_cast<int64>(floorf(in_y));
                    const int64 bottom_y_index = std::min(static_cast<int64>(ceilf(in_y)), (st.in_height - 1));
                    const float y_lerp = in_y - top_y_index;
                    for (int x = 0; x < st.out_width; ++x) {
                        const float in_x = x * st.width_scale;
                        const int64 left_x_index = static_cast<int64>(floorf(in_x));
                        const int64 right_x_index = std::min(static_cast<int64>(ceilf(in_x)), (st.in_width - 1));
                        const float x_lerp = in_x - left_x_index;
                        for (int c = 0; c < st.channels; ++c) {
                            const float front_top_left(input_data(b, front_z_index, top_y_index, left_x_index, c));
                            const float front_top_right(input_data(b, front_z_index, top_y_index, right_x_index, c));
                            const float front_bottom_left(input_data(b, front_z_index, bottom_y_index, left_x_index, c));
                            const float front_bottom_right(input_data(b, front_z_index, bottom_y_index, right_x_index, c));
                            const float back_top_left(input_data(b, back_z_index, top_y_index, left_x_index, c));
                            const float back_top_right(input_data(b, back_z_index, top_y_index, right_x_index, c));
                            const float back_bottom_left(input_data(b, back_z_index, bottom_y_index, left_x_index, c));
                            const float back_bottom_right(input_data(b, back_z_index, bottom_y_index, right_x_index, c));
                            
                            const float front_top = front_top_left + (front_top_right - front_top_left) * x_lerp;
                            const float front_bottom = front_bottom_left + (front_bottom_right - front_bottom_left) * x_lerp;
                            const float back_top = back_top_left + (back_top_right - back_top_left) * x_lerp;
                            const float back_bottom = back_bottom_left + (back_bottom_right - back_bottom_left) * x_lerp;

                            const float front = front_top + (front_bottom - front_top) * y_lerp;
                            const float back = back_top + (back_bottom - back_top) * y_lerp;

                            output_data(b, z, y, x, c) = front + (back - front) * z_lerp;
                        }
                    }
                }
            }
        }
    }
private:
    bool align_corners_;
};

template <typename Device, typename T>
class ResizeBilinear3DOpGrad : public OpKernel {
public:
    explicit ResizeBilinear3DOpGrad(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    }

    void Compute(OpKernelContext* context) override {
        ImageResizerState3D st(align_corners_);
        st.ValidateAndCreateOutput(context);
        if (!context->status().ok()) return;

        typename TTypes<float, 5>::ConstTensor grad_wrt_output = context->input(0).tensor<float, 5>();
        typename TTypes<T, 5>::Tensor grad_wrt_input = st.output->tensor<T, 5>();

        for (int b = 0; b < st.batch_size; ++b) {
            for (int z = 0; z < st.in_depth; ++z) {
                const float in_z = float(z) / st.depth_scale;
                const int64 front_z_index = static_cast<int64>(floorf(in_z));
                const int64 back_z_index = std::min(static_cast<int64>(ceilf(in_z)), (st.out_depth - 1));
                const float z_lerp = in_z - front_z_index;
                for (int y = 0; y < st.in_height; ++y) {
                    const float in_y = float(y) / st.height_scale;
                    const int64 top_y_index = static_cast<int64>(floorf(in_y));
                    const int64 bottom_y_index = std::min(static_cast<int64>(ceilf(in_y)), (st.out_height - 1));
                    const float y_lerp = in_y - top_y_index;
                    for (int x = 0; x < st.in_width; ++x) {
                        const float in_x = float(x) / st.width_scale;
                        const int64 left_x_index = static_cast<int64>(floorf(in_x));
                        const int64 right_x_index = std::min(static_cast<int64>(ceilf(in_x)), (st.out_width - 1));
                        const float x_lerp = in_x - left_x_index;
                        for (int c = 0; c < st.channels; ++c) {
                            const float g(grad_wrt_output(b, z, y, x, c));

                            grad_wrt_input(b, front_z_index, top_y_index, left_x_index, c) += 
                                T(g * (1 - z_lerp) * (1 - y_lerp) * (1 - x_lerp));
                            grad_wrt_input(b, front_z_index, top_y_index, right_x_index, c) +=
                                T(g * (1 - z_lerp) * (1 - y_lerp) * x_lerp);
                            grad_wrt_input(b, front_z_index, bottom_y_index, left_x_index, c) += 
                                T(g * (1 - z_lerp) * y_lerp * (1 - x_lerp));
                            grad_wrt_input(b, front_z_index, bottom_y_index, right_x_index, c) += 
                                T(g * (1 - z_lerp) * y_lerp * x_lerp);
                            grad_wrt_input(b, back_z_index, top_y_index, left_x_index, c) += 
                                T(g * z_lerp * (1 - y_lerp) * (1 - x_lerp));
                            grad_wrt_input(b, back_z_index, top_y_index, right_x_index, c) +=
                                T(g * z_lerp * (1 - y_lerp) * x_lerp);
                            grad_wrt_input(b, back_z_index, bottom_y_index, left_x_index, c) += 
                                T(g * z_lerp * y_lerp * (1 - x_lerp));
                            grad_wrt_input(b, back_z_index, bottom_y_index, right_x_index, c) += 
                                T(g * z_lerp * y_lerp * x_lerp);
                        }
                    }
                }
            }
        }
    }

private:
    bool align_corners_;
};

#define REGISTER_KERNEL(T) \
    REGISTER_KERNEL_BUILDER(Name("ResizeBilinear3D") \
        .Device(DEVICE_CPU) \
        .TypeConstraint<T>("T") \
        .HostMemory("size"), \
        ResizeBilinear3DOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_GRAD_KERNEL(T) \
  REGISTER_KERNEL_BUILDER(Name("ResizeBilinear3DGrad") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T") \
    .HostMemory("input_size"), \
    ResizeBilinear3DOpGrad<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GRAD_KERNEL);

#undef REGISTER_GRAD_KERNEL

}  // namespace tensorflow
