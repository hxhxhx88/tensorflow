#ifndef TENSORFLOW_KERNELS_IMAGE_RESIZER_STATE_3D_H_
#define TENSORFLOW_KERNELS_IMAGE_RESIZER_STATE_3D_H_

#define EIGEN_USE_THREADS

#include <math.h>
#include <algorithm>
#include <array>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/image_resizer_state.h"

namespace tensorflow {

struct ImageResizerState3D {
    explicit ImageResizerState3D(bool align_corners) : align_corners_(align_corners) {}

    void ValidateAndCreateOutput(OpKernelContext* context) {
        //  context->input[0] is the `input`
        //  context->input[1] is the `output_size`, which is a 1-dimensional vector with 3 elements [depth, height, width]
        //  `input` should be a 5-dimentional tensor with shape [batch_size, depth, height, width, channel]
        const Tensor& input = context->input(0);

        OP_REQUIRES(context, input.dims() == 5, errors::InvalidArgument("input must be 5-dimensional", input.shape().DebugString()));

        const Tensor& shape_t = context->input(1);
        OP_REQUIRES(context, shape_t.dims() == 1, errors::InvalidArgument("shape_t must be 1-dimensional", shape_t.shape().DebugString()));
        OP_REQUIRES(context, shape_t.NumElements() == 3, errors::InvalidArgument("shape_t must have three elements", shape_t.shape().DebugString()));

        auto Svec = shape_t.vec<int32>();
        
        batch_size = input.dim_size(0);
        in_depth = static_cast<int32>(input.dim_size(1));
        in_height = static_cast<int32>(input.dim_size(2));
        in_width = static_cast<int32>(input.dim_size(3));
        channels = input.dim_size(4);

        out_depth = internal::SubtleMustCopy(Svec(0));
        out_height = internal::SubtleMustCopy(Svec(1));
        out_width = internal::SubtleMustCopy(Svec(2));
        
        OP_REQUIRES(
            context,
            FastBoundsCheck(in_depth, std::numeric_limits<int32>::max()) &&
            FastBoundsCheck(in_height, std::numeric_limits<int32>::max()) &&
            FastBoundsCheck(in_width, std::numeric_limits<int32>::max()),
            errors::InvalidArgument("input sizes must be between 0 and max int32")
        );
        
        OP_REQUIRES(
            context, 
            out_depth > 0 && out_height > 0 && out_width > 0,
            errors::InvalidArgument("output dimensions must be positive")
        );

        OP_REQUIRES(
            context, 
            channels > 0, 
            errors::InvalidArgument("image must have at least one channel.")
        );

        OP_REQUIRES(
            context, 
            in_depth > 0 && in_height > 0 && in_width > 0,
            errors::InvalidArgument("image must be of non-zero size.")
        )

        OP_REQUIRES_OK(
            context,
            context->allocate_output(0, TensorShape({batch_size, out_depth, out_height, out_width, channels}), &output)
        );

        //  Note: the scale is input_size / output_size, i.e. the factor 
        //  transforming the output back to input.
        depth_scale = CalculateResizeScale(in_depth, out_depth, align_corners_);
        height_scale = CalculateResizeScale(in_height, out_height, align_corners_);
        width_scale = CalculateResizeScale(in_width, out_width, align_corners_);
    }

    int64 batch_size;
    int64 channels;
    int64 out_height;
    int64 out_width;
    int64 out_depth;
    int64 in_height;
    int64 in_width;
    int64 in_depth;
    float height_scale;
    float width_scale;
    float depth_scale;
    Tensor* output;

private:
    bool align_corners_;
};

//  Unlike the implementation in 2D case, we do not create a 
//  ImageResizerGradientState3D struct here, since in fact it is the same thing 
//  as ImageResizerState3D.
//  In ImageResizerState3D, the context (parameters) are:
//      0: input_image, a 5-D tensor with shape [b, d, h, w, c]
//      1: output_shape, a 3-D tensor with shape [d, h, w]
//  In ImageResizerGradientState3D, the context are:
//      0: backprop_grad, a 5-D tensor with shape [b, d, h, w, c]
//      1: input_shape, a 3-D tensor with shape [d, h, w]
//  As a result, we just need to utilize the in_* and out_* properties as
//  different purposes.

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IMAGE_RESIZER_STATE_3D_H_