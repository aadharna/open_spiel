��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
p
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( 
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MatrixDiagV3
diagonal"T
k
num_rows
num_cols
padding_value"T
output"T"	
Ttype"Q
alignstring
RIGHT_LEFT:2
0
LEFT_RIGHT
RIGHT_LEFT	LEFT_LEFTRIGHT_RIGHT
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
|
value_targets/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namevalue_targets/bias
u
&value_targets/bias/Read/ReadVariableOpReadVariableOpvalue_targets/bias*
_output_shapes
:*
dtype0
�
value_targets/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_namevalue_targets/kernel
~
(value_targets/kernel/Read/ReadVariableOpReadVariableOpvalue_targets/kernel*
_output_shapes
:	�*
dtype0
o
flat_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameflat_0/bias
h
flat_0/bias/Read/ReadVariableOpReadVariableOpflat_0/bias*
_output_shapes	
:�*
dtype0
w
flat_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_nameflat_0/kernel
p
!flat_0/kernel/Read/ReadVariableOpReadVariableOpflat_0/kernel*
_output_shapes
:	�*
dtype0
|
policy_logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namepolicy_logits/bias
u
&policy_logits/bias/Read/ReadVariableOpReadVariableOppolicy_logits/bias*
_output_shapes
:*
dtype0
�
policy_logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_namepolicy_logits/kernel
}
(policy_logits/kernel/Read/ReadVariableOpReadVariableOppolicy_logits/kernel*
_output_shapes

:*
dtype0
�
!flat_batch_norm_0/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!flat_batch_norm_0/moving_variance
�
5flat_batch_norm_0/moving_variance/Read/ReadVariableOpReadVariableOp!flat_batch_norm_0/moving_variance*
_output_shapes
:*
dtype0
�
flat_batch_norm_0/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameflat_batch_norm_0/moving_mean
�
1flat_batch_norm_0/moving_mean/Read/ReadVariableOpReadVariableOpflat_batch_norm_0/moving_mean*
_output_shapes
:*
dtype0
�
flat_batch_norm_0/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameflat_batch_norm_0/beta
}
*flat_batch_norm_0/beta/Read/ReadVariableOpReadVariableOpflat_batch_norm_0/beta*
_output_shapes
:*
dtype0
�
flat_batch_norm_0/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameflat_batch_norm_0/gamma

+flat_batch_norm_0/gamma/Read/ReadVariableOpReadVariableOpflat_batch_norm_0/gamma*
_output_shapes
:*
dtype0
�
!conv_batch_norm_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!conv_batch_norm_1/moving_variance
�
5conv_batch_norm_1/moving_variance/Read/ReadVariableOpReadVariableOp!conv_batch_norm_1/moving_variance*
_output_shapes
:*
dtype0
�
conv_batch_norm_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameconv_batch_norm_1/moving_mean
�
1conv_batch_norm_1/moving_mean/Read/ReadVariableOpReadVariableOpconv_batch_norm_1/moving_mean*
_output_shapes
:*
dtype0
�
conv_batch_norm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv_batch_norm_1/beta
}
*conv_batch_norm_1/beta/Read/ReadVariableOpReadVariableOpconv_batch_norm_1/beta*
_output_shapes
:*
dtype0
�
conv_batch_norm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_batch_norm_1/gamma

+conv_batch_norm_1/gamma/Read/ReadVariableOpReadVariableOpconv_batch_norm_1/gamma*
_output_shapes
:*
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
:*
dtype0
v
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameconv_1/kernel
o
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*
_output_shapes

:*
dtype0
�
!conv_batch_norm_0/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!conv_batch_norm_0/moving_variance
�
5conv_batch_norm_0/moving_variance/Read/ReadVariableOpReadVariableOp!conv_batch_norm_0/moving_variance*
_output_shapes
:*
dtype0
�
conv_batch_norm_0/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameconv_batch_norm_0/moving_mean
�
1conv_batch_norm_0/moving_mean/Read/ReadVariableOpReadVariableOpconv_batch_norm_0/moving_mean*
_output_shapes
:*
dtype0
�
conv_batch_norm_0/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameconv_batch_norm_0/beta
}
*conv_batch_norm_0/beta/Read/ReadVariableOpReadVariableOpconv_batch_norm_0/beta*
_output_shapes
:*
dtype0
�
conv_batch_norm_0/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv_batch_norm_0/gamma

+conv_batch_norm_0/gamma/Read/ReadVariableOpReadVariableOpconv_batch_norm_0/gamma*
_output_shapes
:*
dtype0
n
conv_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_0/bias
g
conv_0/bias/Read/ReadVariableOpReadVariableOpconv_0/bias*
_output_shapes
:*
dtype0
v
conv_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameconv_0/kernel
o
!conv_0/kernel/Read/ReadVariableOpReadVariableOpconv_0/kernel*
_output_shapes

:*
dtype0
�
serving_default_environmentPlaceholder*A
_output_shapes/
-:+���������������������������*
dtype0*6
shape-:+���������������������������
p
serving_default_statePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_environmentserving_default_stateconv_0/kernelconv_0/bias!conv_batch_norm_0/moving_varianceconv_batch_norm_0/gammaconv_batch_norm_0/moving_meanconv_batch_norm_0/betaconv_1/kernelconv_1/bias!conv_batch_norm_1/moving_varianceconv_batch_norm_1/gammaconv_batch_norm_1/moving_meanconv_batch_norm_1/betapolicy_logits/kernelpolicy_logits/bias!flat_batch_norm_0/moving_varianceflat_batch_norm_0/gammaflat_batch_norm_0/moving_meanflat_batch_norm_0/betaflat_0/kernelflat_0/biasvalue_targets/kernelvalue_targets/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:������������������:���������*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_84478

NoOpNoOp
�k
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�j
value�jB�j B�j
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-0
layer-11
layer_with_weights-1
layer-12
layer_with_weights-2
layer-13
layer_with_weights-3
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-4
layer-19
layer_with_weights-5
layer-20
layer-21
layer_with_weights-6
layer-22
layer-23
layer-24
layer_with_weights-7
layer-25
layer-26
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_default_save_signature
#	optimizer
$loss
%
signatures*
* 
* 

&	keras_api* 

'	keras_api* 
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 

.	keras_api* 

/	keras_api* 

0	keras_api* 

1	keras_api* 

2	keras_api* 

3	keras_api* 
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:
activation
;aggregation
<W
=b*
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance*
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O
activation
Paggregation
QW
Rb*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance*

^	keras_api* 

_	keras_api* 

`	keras_api* 
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance*
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
x
activation
yaggregation
zW
{b*

|	keras_api* 
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*

�	keras_api* 

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
<0
=1
E2
F3
G4
H5
Q6
R7
Z8
[9
\10
]11
n12
o13
p14
q15
z16
{17
�18
�19
�20
�21*
~
<0
=1
E2
F3
Q4
R5
Z6
[7
n8
o9
z10
{11
�12
�13
�14
�15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
S
�
_variables
�_iterations
�_learning_rate
�_update_step_xla*
* 

�serving_default* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 

<0
=1*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
XR
VARIABLE_VALUEconv_0/kernel1layer_with_weights-0/W/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv_0/bias1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUE*
 
E0
F1
G2
H3*

E0
F1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
f`
VARIABLE_VALUEconv_batch_norm_0/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEconv_batch_norm_0/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEconv_batch_norm_0/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE!conv_batch_norm_0/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
XR
VARIABLE_VALUEconv_1/kernel1layer_with_weights-2/W/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv_1/bias1layer_with_weights-2/b/.ATTRIBUTES/VARIABLE_VALUE*
 
Z0
[1
\2
]3*

Z0
[1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
f`
VARIABLE_VALUEconv_batch_norm_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEconv_batch_norm_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEconv_batch_norm_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE!conv_batch_norm_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
 
n0
o1
p2
q3*

n0
o1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
f`
VARIABLE_VALUEflat_batch_norm_0/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEflat_batch_norm_0/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEflat_batch_norm_0/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE!flat_batch_norm_0/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

z0
{1*

z0
{1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
_Y
VARIABLE_VALUEpolicy_logits/kernel1layer_with_weights-5/W/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEpolicy_logits/bias1layer_with_weights-5/b/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEflat_0/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEflat_0/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
d^
VARIABLE_VALUEvalue_targets/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEvalue_targets/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
.
G0
H1
\2
]3
p4
q5*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26*

�0
�1
�2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
:0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

G0
H1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
O0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

\0
]1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

p0
q1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
x0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv_0/kernelconv_0/biasconv_batch_norm_0/gammaconv_batch_norm_0/betaconv_batch_norm_0/moving_mean!conv_batch_norm_0/moving_varianceconv_1/kernelconv_1/biasconv_batch_norm_1/gammaconv_batch_norm_1/betaconv_batch_norm_1/moving_mean!conv_batch_norm_1/moving_varianceflat_batch_norm_0/gammaflat_batch_norm_0/betaflat_batch_norm_0/moving_mean!flat_batch_norm_0/moving_variancepolicy_logits/kernelpolicy_logits/biasflat_0/kernelflat_0/biasvalue_targets/kernelvalue_targets/bias	iterationlearning_ratetotal_2count_2total_1count_1totalcountConst*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_85911
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_0/kernelconv_0/biasconv_batch_norm_0/gammaconv_batch_norm_0/betaconv_batch_norm_0/moving_mean!conv_batch_norm_0/moving_varianceconv_1/kernelconv_1/biasconv_batch_norm_1/gammaconv_batch_norm_1/betaconv_batch_norm_1/moving_mean!conv_batch_norm_1/moving_varianceflat_batch_norm_0/gammaflat_batch_norm_0/betaflat_batch_norm_0/moving_mean!flat_batch_norm_0/moving_variancepolicy_logits/kernelpolicy_logits/biasflat_0/kernelflat_0/biasvalue_targets/kernelvalue_targets/bias	iterationlearning_ratetotal_2count_2total_1count_1totalcount**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_86011��
�
�
1__inference_flat_batch_norm_0_layer_call_fn_85481

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_83373o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�m
�	
A__inference_gnn_63_layer_call_and_return_conditional_losses_84160

inputs
inputs_1
conv_0_84089:
conv_0_84091:%
conv_batch_norm_0_84094:%
conv_batch_norm_0_84096:%
conv_batch_norm_0_84098:%
conv_batch_norm_0_84100:
conv_1_84103:
conv_1_84105:%
conv_batch_norm_1_84108:%
conv_batch_norm_1_84110:%
conv_batch_norm_1_84112:%
conv_batch_norm_1_84114:%
policy_logits_84127:!
policy_logits_84129:%
flat_batch_norm_0_84132:%
flat_batch_norm_0_84134:%
flat_batch_norm_0_84136:%
flat_batch_norm_0_84138:
flat_0_84147:	�
flat_0_84149:	�&
value_targets_84153:	�!
value_targets_84155:
identity

identity_1��conv_0/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�)conv_batch_norm_0/StatefulPartitionedCall�)conv_batch_norm_1/StatefulPartitionedCall�flat_0/StatefulPartitionedCall�)flat_batch_norm_0/StatefulPartitionedCall�%policy_logits/StatefulPartitionedCall�%value_targets/StatefulPartitionedCall_
tf.compat.v1.shape_66/ShapeShapeinputs*
T0*
_output_shapes
::��_
tf.cast_131/CastCastinputs_1*

DstT0*

SrcT0*#
_output_shapes
:����������
reshape_65/PartitionedCallPartitionedCalltf.cast_131/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_65_layer_call_and_return_conditional_losses_83439z
0tf.__operators__.getitem_324/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*tf.__operators__.getitem_324/strided_sliceStridedSlice$tf.compat.v1.shape_66/Shape:output:09tf.__operators__.getitem_324/strided_slice/stack:output:0;tf.__operators__.getitem_324/strided_slice/stack_1:output:0;tf.__operators__.getitem_324/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
tf.one_hot_66/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
tf.one_hot_66/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
tf.one_hot_66/one_hotOneHot#reshape_65/PartitionedCall:output:03tf.__operators__.getitem_324/strided_slice:output:0'tf.one_hot_66/one_hot/on_value:output:0(tf.one_hot_66/one_hot/off_value:output:0*
TI0*
T0*4
_output_shapes"
 :�������������������
0tf.__operators__.getitem_325/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_325/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_325/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_325/strided_sliceStridedSliceinputs9tf.__operators__.getitem_325/strided_slice/stack:output:0;tf.__operators__.getitem_325/strided_slice/stack_1:output:0;tf.__operators__.getitem_325/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask�
0tf.__operators__.getitem_326/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_326/strided_sliceStridedSliceinputs9tf.__operators__.getitem_326/strided_slice/stack:output:0;tf.__operators__.getitem_326/strided_slice/stack_1:output:0;tf.__operators__.getitem_326/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask~
)tf.compat.v1.transpose_130/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_130/transpose	Transposetf.one_hot_66/one_hot:output:02tf.compat.v1.transpose_130/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
conv_0/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:0(tf.compat.v1.transpose_130/transpose:y:0conv_0_84089conv_0_84091*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_0_layer_call_and_return_conditional_losses_83762�
)conv_batch_norm_0/StatefulPartitionedCallStatefulPartitionedCall'conv_0/StatefulPartitionedCall:output:0conv_batch_norm_0_84094conv_batch_norm_0_84096conv_batch_norm_0_84098conv_batch_norm_0_84100*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_83229�
conv_1/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:02conv_batch_norm_0/StatefulPartitionedCall:output:0conv_1_84103conv_1_84105*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_83816�
)conv_batch_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_batch_norm_1_84108conv_batch_norm_1_84110conv_batch_norm_1_84112conv_batch_norm_1_84114*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_83311~
)tf.compat.v1.transpose_131/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_131/transpose	Transpose2conv_batch_norm_1/StatefulPartitionedCall:output:02tf.compat.v1.transpose_131/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
tf.math.multiply_65/MulMul(tf.compat.v1.transpose_131/transpose:y:0tf.one_hot_66/one_hot:output:0*
T0*4
_output_shapes"
 :�������������������
legals_mask/PartitionedCallPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:0tf.one_hot_66/one_hot:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_legals_mask_layer_call_and_return_conditional_losses_83582v
+tf.math.reduce_sum_64/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_64/SumSumtf.math.multiply_65/Mul:z:04tf.math.reduce_sum_64/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:����������
0tf.__operators__.getitem_327/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_327/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_327/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_327/strided_sliceStridedSlice$legals_mask/PartitionedCall:output:09tf.__operators__.getitem_327/strided_slice/stack:output:0;tf.__operators__.getitem_327/strided_slice/stack_1:output:0;tf.__operators__.getitem_327/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_mask�
%policy_logits/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:02conv_batch_norm_1/StatefulPartitionedCall:output:0policy_logits_84127policy_logits_84129*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_logits_layer_call_and_return_conditional_losses_83880�
)flat_batch_norm_0/StatefulPartitionedCallStatefulPartitionedCall"tf.math.reduce_sum_64/Sum:output:0flat_batch_norm_0_84132flat_batch_norm_0_84134flat_batch_norm_0_84136flat_batch_norm_0_84138*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_83393�
0tf.__operators__.getitem_328/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_328/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_328/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_328/strided_sliceStridedSlice.policy_logits/StatefulPartitionedCall:output:09tf.__operators__.getitem_328/strided_slice/stack:output:0;tf.__operators__.getitem_328/strided_slice/stack_1:output:0;tf.__operators__.getitem_328/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_maska
tf.math.greater_59/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater_59/GreaterGreater3tf.__operators__.getitem_327/strided_slice:output:0%tf.math.greater_59/Greater/y:output:0*
T0*0
_output_shapes
:�������������������
flat_0/StatefulPartitionedCallStatefulPartitionedCall2flat_batch_norm_0/StatefulPartitionedCall:output:0flat_0_84147flat_0_84149*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_flat_0_layer_call_and_return_conditional_losses_83662�
policy_targets/PartitionedCallPartitionedCall3tf.__operators__.getitem_328/strided_slice:output:0tf.math.greater_59/Greater:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_policy_targets_layer_call_and_return_conditional_losses_83678�
%value_targets/StatefulPartitionedCallStatefulPartitionedCall'flat_0/StatefulPartitionedCall:output:0value_targets_84153value_targets_84155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_value_targets_layer_call_and_return_conditional_losses_83691}
IdentityIdentity.value_targets/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������

Identity_1Identity'policy_targets/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�������������������
NoOpNoOp^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall*^conv_batch_norm_0/StatefulPartitionedCall*^conv_batch_norm_1/StatefulPartitionedCall^flat_0/StatefulPartitionedCall*^flat_batch_norm_0/StatefulPartitionedCall&^policy_logits/StatefulPartitionedCall&^value_targets/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2V
)conv_batch_norm_0/StatefulPartitionedCall)conv_batch_norm_0/StatefulPartitionedCall2V
)conv_batch_norm_1/StatefulPartitionedCall)conv_batch_norm_1/StatefulPartitionedCall2@
flat_0/StatefulPartitionedCallflat_0/StatefulPartitionedCall2V
)flat_batch_norm_0/StatefulPartitionedCall)flat_batch_norm_0/StatefulPartitionedCall2N
%policy_logits/StatefulPartitionedCall%policy_logits/StatefulPartitionedCall2N
%value_targets/StatefulPartitionedCall%value_targets/StatefulPartitionedCall:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�%
�
A__inference_conv_1_layer_call_and_return_conditional_losses_85331
inputs_0
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_238/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_238/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�%
�
H__inference_policy_logits_layer_call_and_return_conditional_losses_83880

inputs
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_239/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_239/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:\X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�

�
H__inference_value_targets_layer_call_and_return_conditional_losses_83691

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
a
E__inference_reshape_65_layer_call_and_return_conditional_losses_85087

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:���������:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
&__inference_conv_1_layer_call_fn_85291
inputs_0
inputs_1
inputs_2
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_83816|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�
F
*__inference_reshape_65_layer_call_fn_85075

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_65_layer_call_and_return_conditional_losses_83439`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:���������:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_83229

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�%
�
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_83291

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
1__inference_conv_batch_norm_1_layer_call_fn_85397

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_83311|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�	
p
F__inference_legals_mask_layer_call_and_return_conditional_losses_83582

inputs
inputs_1
identityc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_1	Transposeinputs_1transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������v
MatMulBatchMatMulV2transpose:y:0transpose_1:y:0*
T0*4
_output_shapes"
 :������������������R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��d
IdentityIdentityMatMul:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:'���������������������������:������������������:\X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
u
I__inference_policy_targets_layer_call_and_return_conditional_losses_85706
inputs_0
inputs_1

identityK
ShapeShapeinputs_0*
T0*
_output_shapes
::��O

Fill/valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ��l
FillFillShape:output:0Fill/value:output:0*
T0*0
_output_shapes
:������������������r
SelectV2SelectV2inputs_1inputs_0Fill:output:0*
T0*0
_output_shapes
:������������������`
SoftmaxSoftmaxSelectV2:output:0*
T0*0
_output_shapes
:������������������b
IdentityIdentitySoftmax:softmax:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������:������������������:ZV
0
_output_shapes
:������������������
"
_user_specified_name
inputs_1:Z V
0
_output_shapes
:������������������
"
_user_specified_name
inputs_0
�m
�	
A__inference_gnn_63_layer_call_and_return_conditional_losses_83914
environment	
state
conv_0_83763:
conv_0_83765:%
conv_batch_norm_0_83768:%
conv_batch_norm_0_83770:%
conv_batch_norm_0_83772:%
conv_batch_norm_0_83774:
conv_1_83817:
conv_1_83819:%
conv_batch_norm_1_83822:%
conv_batch_norm_1_83824:%
conv_batch_norm_1_83826:%
conv_batch_norm_1_83828:%
policy_logits_83881:!
policy_logits_83883:%
flat_batch_norm_0_83886:%
flat_batch_norm_0_83888:%
flat_batch_norm_0_83890:%
flat_batch_norm_0_83892:
flat_0_83901:	�
flat_0_83903:	�&
value_targets_83907:	�!
value_targets_83909:
identity

identity_1��conv_0/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�)conv_batch_norm_0/StatefulPartitionedCall�)conv_batch_norm_1/StatefulPartitionedCall�flat_0/StatefulPartitionedCall�)flat_batch_norm_0/StatefulPartitionedCall�%policy_logits/StatefulPartitionedCall�%value_targets/StatefulPartitionedCalld
tf.compat.v1.shape_66/ShapeShapeenvironment*
T0*
_output_shapes
::��\
tf.cast_131/CastCaststate*

DstT0*

SrcT0*#
_output_shapes
:����������
reshape_65/PartitionedCallPartitionedCalltf.cast_131/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_65_layer_call_and_return_conditional_losses_83439z
0tf.__operators__.getitem_324/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*tf.__operators__.getitem_324/strided_sliceStridedSlice$tf.compat.v1.shape_66/Shape:output:09tf.__operators__.getitem_324/strided_slice/stack:output:0;tf.__operators__.getitem_324/strided_slice/stack_1:output:0;tf.__operators__.getitem_324/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
tf.one_hot_66/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
tf.one_hot_66/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
tf.one_hot_66/one_hotOneHot#reshape_65/PartitionedCall:output:03tf.__operators__.getitem_324/strided_slice:output:0'tf.one_hot_66/one_hot/on_value:output:0(tf.one_hot_66/one_hot/off_value:output:0*
TI0*
T0*4
_output_shapes"
 :�������������������
0tf.__operators__.getitem_325/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_325/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_325/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_325/strided_sliceStridedSliceenvironment9tf.__operators__.getitem_325/strided_slice/stack:output:0;tf.__operators__.getitem_325/strided_slice/stack_1:output:0;tf.__operators__.getitem_325/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask�
0tf.__operators__.getitem_326/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_326/strided_sliceStridedSliceenvironment9tf.__operators__.getitem_326/strided_slice/stack:output:0;tf.__operators__.getitem_326/strided_slice/stack_1:output:0;tf.__operators__.getitem_326/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask~
)tf.compat.v1.transpose_130/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_130/transpose	Transposetf.one_hot_66/one_hot:output:02tf.compat.v1.transpose_130/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
conv_0/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:0(tf.compat.v1.transpose_130/transpose:y:0conv_0_83763conv_0_83765*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_0_layer_call_and_return_conditional_losses_83762�
)conv_batch_norm_0/StatefulPartitionedCallStatefulPartitionedCall'conv_0/StatefulPartitionedCall:output:0conv_batch_norm_0_83768conv_batch_norm_0_83770conv_batch_norm_0_83772conv_batch_norm_0_83774*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_83229�
conv_1/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:02conv_batch_norm_0/StatefulPartitionedCall:output:0conv_1_83817conv_1_83819*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_83816�
)conv_batch_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_batch_norm_1_83822conv_batch_norm_1_83824conv_batch_norm_1_83826conv_batch_norm_1_83828*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_83311~
)tf.compat.v1.transpose_131/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_131/transpose	Transpose2conv_batch_norm_1/StatefulPartitionedCall:output:02tf.compat.v1.transpose_131/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
tf.math.multiply_65/MulMul(tf.compat.v1.transpose_131/transpose:y:0tf.one_hot_66/one_hot:output:0*
T0*4
_output_shapes"
 :�������������������
legals_mask/PartitionedCallPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:0tf.one_hot_66/one_hot:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_legals_mask_layer_call_and_return_conditional_losses_83582v
+tf.math.reduce_sum_64/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_64/SumSumtf.math.multiply_65/Mul:z:04tf.math.reduce_sum_64/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:����������
0tf.__operators__.getitem_327/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_327/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_327/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_327/strided_sliceStridedSlice$legals_mask/PartitionedCall:output:09tf.__operators__.getitem_327/strided_slice/stack:output:0;tf.__operators__.getitem_327/strided_slice/stack_1:output:0;tf.__operators__.getitem_327/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_mask�
%policy_logits/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:02conv_batch_norm_1/StatefulPartitionedCall:output:0policy_logits_83881policy_logits_83883*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_logits_layer_call_and_return_conditional_losses_83880�
)flat_batch_norm_0/StatefulPartitionedCallStatefulPartitionedCall"tf.math.reduce_sum_64/Sum:output:0flat_batch_norm_0_83886flat_batch_norm_0_83888flat_batch_norm_0_83890flat_batch_norm_0_83892*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_83393�
0tf.__operators__.getitem_328/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_328/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_328/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_328/strided_sliceStridedSlice.policy_logits/StatefulPartitionedCall:output:09tf.__operators__.getitem_328/strided_slice/stack:output:0;tf.__operators__.getitem_328/strided_slice/stack_1:output:0;tf.__operators__.getitem_328/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_maska
tf.math.greater_59/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater_59/GreaterGreater3tf.__operators__.getitem_327/strided_slice:output:0%tf.math.greater_59/Greater/y:output:0*
T0*0
_output_shapes
:�������������������
flat_0/StatefulPartitionedCallStatefulPartitionedCall2flat_batch_norm_0/StatefulPartitionedCall:output:0flat_0_83901flat_0_83903*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_flat_0_layer_call_and_return_conditional_losses_83662�
policy_targets/PartitionedCallPartitionedCall3tf.__operators__.getitem_328/strided_slice:output:0tf.math.greater_59/Greater:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_policy_targets_layer_call_and_return_conditional_losses_83678�
%value_targets/StatefulPartitionedCallStatefulPartitionedCall'flat_0/StatefulPartitionedCall:output:0value_targets_83907value_targets_83909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_value_targets_layer_call_and_return_conditional_losses_83691}
IdentityIdentity.value_targets/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������

Identity_1Identity'policy_targets/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�������������������
NoOpNoOp^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall*^conv_batch_norm_0/StatefulPartitionedCall*^conv_batch_norm_1/StatefulPartitionedCall^flat_0/StatefulPartitionedCall*^flat_batch_norm_0/StatefulPartitionedCall&^policy_logits/StatefulPartitionedCall&^value_targets/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2V
)conv_batch_norm_0/StatefulPartitionedCall)conv_batch_norm_0/StatefulPartitionedCall2V
)conv_batch_norm_1/StatefulPartitionedCall)conv_batch_norm_1/StatefulPartitionedCall2@
flat_0/StatefulPartitionedCallflat_0/StatefulPartitionedCall2V
)flat_batch_norm_0/StatefulPartitionedCall)flat_batch_norm_0/StatefulPartitionedCall2N
%policy_logits/StatefulPartitionedCall%policy_logits/StatefulPartitionedCall2N
%value_targets/StatefulPartitionedCall%value_targets/StatefulPartitionedCall:JF
#
_output_shapes
:���������

_user_specified_namestate:n j
A
_output_shapes/
-:+���������������������������
%
_user_specified_nameenvironment
�%
�
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_85528

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_85548

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_gnn_63_layer_call_fn_84582
inputs_0
inputs_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:���������:������������������*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gnn_63_layer_call_and_return_conditional_losses_84160o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������z

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*0
_output_shapes
:������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs_1:k g
A
_output_shapes/
-:+���������������������������
"
_user_specified_name
inputs_0
�
�
-__inference_value_targets_layer_call_fn_85679

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_value_targets_layer_call_and_return_conditional_losses_83691o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
&__inference_conv_1_layer_call_fn_85280
inputs_0
inputs_1
inputs_2
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_83553|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�
s
I__inference_policy_targets_layer_call_and_return_conditional_losses_83678

inputs
inputs_1

identityI
ShapeShapeinputs*
T0*
_output_shapes
::��O

Fill/valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ��l
FillFillShape:output:0Fill/value:output:0*
T0*0
_output_shapes
:������������������p
SelectV2SelectV2inputs_1inputsFill:output:0*
T0*0
_output_shapes
:������������������`
SoftmaxSoftmaxSelectV2:output:0*
T0*0
_output_shapes
:������������������b
IdentityIdentitySoftmax:softmax:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������:������������������:XT
0
_output_shapes
:������������������
 
_user_specified_nameinputs:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_83393

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
A__inference_conv_0_layer_call_and_return_conditional_losses_83762

inputs
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_237/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_237/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:\X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_85269

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�%
�
A__inference_conv_1_layer_call_and_return_conditional_losses_83553

inputs
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_238/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_238/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:\X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
��
�
A__inference_gnn_63_layer_call_and_return_conditional_losses_85070
inputs_0
inputs_19
'conv_0_matmul_1_readvariableop_resource:2
$conv_0_add_1_readvariableop_resource:A
3conv_batch_norm_0_batchnorm_readvariableop_resource:E
7conv_batch_norm_0_batchnorm_mul_readvariableop_resource:C
5conv_batch_norm_0_batchnorm_readvariableop_1_resource:C
5conv_batch_norm_0_batchnorm_readvariableop_2_resource:9
'conv_1_matmul_1_readvariableop_resource:2
$conv_1_add_1_readvariableop_resource:A
3conv_batch_norm_1_batchnorm_readvariableop_resource:E
7conv_batch_norm_1_batchnorm_mul_readvariableop_resource:C
5conv_batch_norm_1_batchnorm_readvariableop_1_resource:C
5conv_batch_norm_1_batchnorm_readvariableop_2_resource:@
.policy_logits_matmul_1_readvariableop_resource:9
+policy_logits_add_1_readvariableop_resource:A
3flat_batch_norm_0_batchnorm_readvariableop_resource:E
7flat_batch_norm_0_batchnorm_mul_readvariableop_resource:C
5flat_batch_norm_0_batchnorm_readvariableop_1_resource:C
5flat_batch_norm_0_batchnorm_readvariableop_2_resource:8
%flat_0_matmul_readvariableop_resource:	�5
&flat_0_biasadd_readvariableop_resource:	�?
,value_targets_matmul_readvariableop_resource:	�;
-value_targets_biasadd_readvariableop_resource:
identity

identity_1��conv_0/MatMul_1/ReadVariableOp�conv_0/add_1/ReadVariableOp�conv_1/MatMul_1/ReadVariableOp�conv_1/add_1/ReadVariableOp�*conv_batch_norm_0/batchnorm/ReadVariableOp�,conv_batch_norm_0/batchnorm/ReadVariableOp_1�,conv_batch_norm_0/batchnorm/ReadVariableOp_2�.conv_batch_norm_0/batchnorm/mul/ReadVariableOp�*conv_batch_norm_1/batchnorm/ReadVariableOp�,conv_batch_norm_1/batchnorm/ReadVariableOp_1�,conv_batch_norm_1/batchnorm/ReadVariableOp_2�.conv_batch_norm_1/batchnorm/mul/ReadVariableOp�flat_0/BiasAdd/ReadVariableOp�flat_0/MatMul/ReadVariableOp�*flat_batch_norm_0/batchnorm/ReadVariableOp�,flat_batch_norm_0/batchnorm/ReadVariableOp_1�,flat_batch_norm_0/batchnorm/ReadVariableOp_2�.flat_batch_norm_0/batchnorm/mul/ReadVariableOp�%policy_logits/MatMul_1/ReadVariableOp�"policy_logits/add_1/ReadVariableOp�$value_targets/BiasAdd/ReadVariableOp�#value_targets/MatMul/ReadVariableOpa
tf.compat.v1.shape_66/ShapeShapeinputs_0*
T0*
_output_shapes
::��_
tf.cast_131/CastCastinputs_1*

DstT0*

SrcT0*#
_output_shapes
:���������b
reshape_65/ShapeShapetf.cast_131/Cast:y:0*
T0*
_output_shapes
::��h
reshape_65/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_65/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_65/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_65/strided_sliceStridedSlicereshape_65/Shape:output:0'reshape_65/strided_slice/stack:output:0)reshape_65/strided_slice/stack_1:output:0)reshape_65/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_65/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_65/Reshape/shapePack!reshape_65/strided_slice:output:0#reshape_65/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
reshape_65/ReshapeReshapetf.cast_131/Cast:y:0!reshape_65/Reshape/shape:output:0*
T0*'
_output_shapes
:���������z
0tf.__operators__.getitem_324/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*tf.__operators__.getitem_324/strided_sliceStridedSlice$tf.compat.v1.shape_66/Shape:output:09tf.__operators__.getitem_324/strided_slice/stack:output:0;tf.__operators__.getitem_324/strided_slice/stack_1:output:0;tf.__operators__.getitem_324/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
tf.one_hot_66/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
tf.one_hot_66/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
tf.one_hot_66/one_hotOneHotreshape_65/Reshape:output:03tf.__operators__.getitem_324/strided_slice:output:0'tf.one_hot_66/one_hot/on_value:output:0(tf.one_hot_66/one_hot/off_value:output:0*
TI0*
T0*4
_output_shapes"
 :�������������������
0tf.__operators__.getitem_325/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_325/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_325/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_325/strided_sliceStridedSliceinputs_09tf.__operators__.getitem_325/strided_slice/stack:output:0;tf.__operators__.getitem_325/strided_slice/stack_1:output:0;tf.__operators__.getitem_325/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask�
0tf.__operators__.getitem_326/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_326/strided_sliceStridedSliceinputs_09tf.__operators__.getitem_326/strided_slice/stack:output:0;tf.__operators__.getitem_326/strided_slice/stack_1:output:0;tf.__operators__.getitem_326/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask~
)tf.compat.v1.transpose_130/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_130/transpose	Transposetf.one_hot_66/one_hot:output:02tf.compat.v1.transpose_130/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������}
conv_0/ShapeShape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��m
conv_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������f
conv_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
conv_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_0/strided_sliceStridedSliceconv_0/Shape:output:0#conv_0/strided_slice/stack:output:0%conv_0/strided_slice/stack_1:output:0%conv_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv_0/Shape_1Shape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��f
conv_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
conv_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������h
conv_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_0/strided_slice_1StridedSliceconv_0/Shape_1:output:0%conv_0/strided_slice_1/stack:output:0'conv_0/strided_slice_1/stack_1:output:0'conv_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
conv_0/eye/MinimumMinimumconv_0/strided_slice:output:0conv_0/strided_slice:output:0*
T0*
_output_shapes
: h
conv_0/eye/concat/values_1Packconv_0/eye/Minimum:z:0*
N*
T0*
_output_shapes
:X
conv_0/eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
conv_0/eye/concatConcatV2conv_0/strided_slice_1:output:0#conv_0/eye/concat/values_1:output:0conv_0/eye/concat/axis:output:0*
N*
T0*
_output_shapes
:Z
conv_0/eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv_0/eye/onesFillconv_0/eye/concat:output:0conv_0/eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������S
conv_0/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : c
conv_0/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������c
conv_0/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������b
conv_0/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_0/eye/diagMatrixDiagV3conv_0/eye/ones:output:0conv_0/eye/diag/k:output:0!conv_0/eye/diag/num_rows:output:0!conv_0/eye/diag/num_cols:output:0&conv_0/eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'����������������������������

conv_0/addAddV23tf.__operators__.getitem_326/strided_slice:output:0conv_0/eye/diag:output:0*
T0*=
_output_shapes+
):'����������������������������
conv_0/norm/mulMul3tf.__operators__.getitem_326/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:0*
T0*=
_output_shapes+
):'���������������������������t
!conv_0/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
conv_0/norm/SumSumconv_0/norm/mul:z:0*conv_0/norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(q
conv_0/norm/SqrtSqrtconv_0/norm/Sum:output:0*
T0*4
_output_shapes"
 :�������������������
conv_0/MatMulBatchMatMulV2conv_0/add:z:0(tf.compat.v1.transpose_130/transpose:y:0*
T0*4
_output_shapes"
 :�������������������
conv_0/MatMul_1/ReadVariableOpReadVariableOp'conv_0_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
conv_0/MatMul_1BatchMatMulV2conv_0/MatMul:output:0&conv_0/MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������|
conv_0/add_1/ReadVariableOpReadVariableOp$conv_0_add_1_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_0/add_1AddV2conv_0/MatMul_1:output:0#conv_0/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������s
conv_0/activation_237/ReluReluconv_0/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
*conv_batch_norm_0/batchnorm/ReadVariableOpReadVariableOp3conv_batch_norm_0_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0f
!conv_batch_norm_0/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv_batch_norm_0/batchnorm/addAddV22conv_batch_norm_0/batchnorm/ReadVariableOp:value:0*conv_batch_norm_0/batchnorm/add/y:output:0*
T0*
_output_shapes
:t
!conv_batch_norm_0/batchnorm/RsqrtRsqrt#conv_batch_norm_0/batchnorm/add:z:0*
T0*
_output_shapes
:�
.conv_batch_norm_0/batchnorm/mul/ReadVariableOpReadVariableOp7conv_batch_norm_0_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_batch_norm_0/batchnorm/mulMul%conv_batch_norm_0/batchnorm/Rsqrt:y:06conv_batch_norm_0/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
!conv_batch_norm_0/batchnorm/mul_1Mul(conv_0/activation_237/Relu:activations:0#conv_batch_norm_0/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :�������������������
,conv_batch_norm_0/batchnorm/ReadVariableOp_1ReadVariableOp5conv_batch_norm_0_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
!conv_batch_norm_0/batchnorm/mul_2Mul4conv_batch_norm_0/batchnorm/ReadVariableOp_1:value:0#conv_batch_norm_0/batchnorm/mul:z:0*
T0*
_output_shapes
:�
,conv_batch_norm_0/batchnorm/ReadVariableOp_2ReadVariableOp5conv_batch_norm_0_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
conv_batch_norm_0/batchnorm/subSub4conv_batch_norm_0/batchnorm/ReadVariableOp_2:value:0%conv_batch_norm_0/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
!conv_batch_norm_0/batchnorm/add_1AddV2%conv_batch_norm_0/batchnorm/mul_1:z:0#conv_batch_norm_0/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������}
conv_1/ShapeShape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��m
conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������f
conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_1/strided_sliceStridedSliceconv_1/Shape:output:0#conv_1/strided_slice/stack:output:0%conv_1/strided_slice/stack_1:output:0%conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv_1/Shape_1Shape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��f
conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������h
conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_1/strided_slice_1StridedSliceconv_1/Shape_1:output:0%conv_1/strided_slice_1/stack:output:0'conv_1/strided_slice_1/stack_1:output:0'conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
conv_1/eye/MinimumMinimumconv_1/strided_slice:output:0conv_1/strided_slice:output:0*
T0*
_output_shapes
: h
conv_1/eye/concat/values_1Packconv_1/eye/Minimum:z:0*
N*
T0*
_output_shapes
:X
conv_1/eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
conv_1/eye/concatConcatV2conv_1/strided_slice_1:output:0#conv_1/eye/concat/values_1:output:0conv_1/eye/concat/axis:output:0*
N*
T0*
_output_shapes
:Z
conv_1/eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv_1/eye/onesFillconv_1/eye/concat:output:0conv_1/eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������S
conv_1/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : c
conv_1/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������c
conv_1/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������b
conv_1/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_1/eye/diagMatrixDiagV3conv_1/eye/ones:output:0conv_1/eye/diag/k:output:0!conv_1/eye/diag/num_rows:output:0!conv_1/eye/diag/num_cols:output:0&conv_1/eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'����������������������������

conv_1/addAddV23tf.__operators__.getitem_326/strided_slice:output:0conv_1/eye/diag:output:0*
T0*=
_output_shapes+
):'����������������������������
conv_1/norm/mulMul3tf.__operators__.getitem_326/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:0*
T0*=
_output_shapes+
):'���������������������������t
!conv_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
conv_1/norm/SumSumconv_1/norm/mul:z:0*conv_1/norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(q
conv_1/norm/SqrtSqrtconv_1/norm/Sum:output:0*
T0*4
_output_shapes"
 :�������������������
conv_1/MatMulBatchMatMulV2conv_1/add:z:0%conv_batch_norm_0/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
conv_1/MatMul_1/ReadVariableOpReadVariableOp'conv_1_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
conv_1/MatMul_1BatchMatMulV2conv_1/MatMul:output:0&conv_1/MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������|
conv_1/add_1/ReadVariableOpReadVariableOp$conv_1_add_1_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_1/add_1AddV2conv_1/MatMul_1:output:0#conv_1/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������s
conv_1/activation_238/ReluReluconv_1/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
*conv_batch_norm_1/batchnorm/ReadVariableOpReadVariableOp3conv_batch_norm_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0f
!conv_batch_norm_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv_batch_norm_1/batchnorm/addAddV22conv_batch_norm_1/batchnorm/ReadVariableOp:value:0*conv_batch_norm_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:t
!conv_batch_norm_1/batchnorm/RsqrtRsqrt#conv_batch_norm_1/batchnorm/add:z:0*
T0*
_output_shapes
:�
.conv_batch_norm_1/batchnorm/mul/ReadVariableOpReadVariableOp7conv_batch_norm_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_batch_norm_1/batchnorm/mulMul%conv_batch_norm_1/batchnorm/Rsqrt:y:06conv_batch_norm_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
!conv_batch_norm_1/batchnorm/mul_1Mul(conv_1/activation_238/Relu:activations:0#conv_batch_norm_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :�������������������
,conv_batch_norm_1/batchnorm/ReadVariableOp_1ReadVariableOp5conv_batch_norm_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
!conv_batch_norm_1/batchnorm/mul_2Mul4conv_batch_norm_1/batchnorm/ReadVariableOp_1:value:0#conv_batch_norm_1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
,conv_batch_norm_1/batchnorm/ReadVariableOp_2ReadVariableOp5conv_batch_norm_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
conv_batch_norm_1/batchnorm/subSub4conv_batch_norm_1/batchnorm/ReadVariableOp_2:value:0%conv_batch_norm_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
!conv_batch_norm_1/batchnorm/add_1AddV2%conv_batch_norm_1/batchnorm/mul_1:z:0#conv_batch_norm_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������~
)tf.compat.v1.transpose_131/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_131/transpose	Transpose%conv_batch_norm_1/batchnorm/add_1:z:02tf.compat.v1.transpose_131/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
tf.math.multiply_65/MulMul(tf.compat.v1.transpose_131/transpose:y:0tf.one_hot_66/one_hot:output:0*
T0*4
_output_shapes"
 :������������������o
legals_mask/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
legals_mask/transpose	Transpose3tf.__operators__.getitem_325/strided_slice:output:0#legals_mask/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������q
legals_mask/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
legals_mask/transpose_1	Transposetf.one_hot_66/one_hot:output:0%legals_mask/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :�������������������
legals_mask/MatMulBatchMatMulV2legals_mask/transpose:y:0legals_mask/transpose_1:y:0*
T0*4
_output_shapes"
 :������������������j
legals_mask/ShapeShapelegals_mask/MatMul:output:0*
T0*
_output_shapes
::��v
+tf.math.reduce_sum_64/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_64/SumSumtf.math.multiply_65/Mul:z:04tf.math.reduce_sum_64/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:����������
0tf.__operators__.getitem_327/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_327/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_327/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_327/strided_sliceStridedSlicelegals_mask/MatMul:output:09tf.__operators__.getitem_327/strided_slice/stack:output:0;tf.__operators__.getitem_327/strided_slice/stack_1:output:0;tf.__operators__.getitem_327/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_mask�
policy_logits/ShapeShape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��t
!policy_logits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������m
#policy_logits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#policy_logits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
policy_logits/strided_sliceStridedSlicepolicy_logits/Shape:output:0*policy_logits/strided_slice/stack:output:0,policy_logits/strided_slice/stack_1:output:0,policy_logits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
policy_logits/Shape_1Shape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��m
#policy_logits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%policy_logits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������o
%policy_logits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
policy_logits/strided_slice_1StridedSlicepolicy_logits/Shape_1:output:0,policy_logits/strided_slice_1/stack:output:0.policy_logits/strided_slice_1/stack_1:output:0.policy_logits/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
policy_logits/eye/MinimumMinimum$policy_logits/strided_slice:output:0$policy_logits/strided_slice:output:0*
T0*
_output_shapes
: v
!policy_logits/eye/concat/values_1Packpolicy_logits/eye/Minimum:z:0*
N*
T0*
_output_shapes
:_
policy_logits/eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
policy_logits/eye/concatConcatV2&policy_logits/strided_slice_1:output:0*policy_logits/eye/concat/values_1:output:0&policy_logits/eye/concat/axis:output:0*
N*
T0*
_output_shapes
:a
policy_logits/eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
policy_logits/eye/onesFill!policy_logits/eye/concat:output:0%policy_logits/eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������Z
policy_logits/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : j
policy_logits/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������j
policy_logits/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������i
$policy_logits/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
policy_logits/eye/diagMatrixDiagV3policy_logits/eye/ones:output:0!policy_logits/eye/diag/k:output:0(policy_logits/eye/diag/num_rows:output:0(policy_logits/eye/diag/num_cols:output:0-policy_logits/eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'����������������������������
policy_logits/addAddV23tf.__operators__.getitem_326/strided_slice:output:0policy_logits/eye/diag:output:0*
T0*=
_output_shapes+
):'����������������������������
policy_logits/norm/mulMul3tf.__operators__.getitem_326/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:0*
T0*=
_output_shapes+
):'���������������������������{
(policy_logits/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
policy_logits/norm/SumSumpolicy_logits/norm/mul:z:01policy_logits/norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(
policy_logits/norm/SqrtSqrtpolicy_logits/norm/Sum:output:0*
T0*4
_output_shapes"
 :�������������������
policy_logits/MatMulBatchMatMulV2policy_logits/add:z:0%conv_batch_norm_1/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
%policy_logits/MatMul_1/ReadVariableOpReadVariableOp.policy_logits_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
policy_logits/MatMul_1BatchMatMulV2policy_logits/MatMul:output:0-policy_logits/MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
"policy_logits/add_1/ReadVariableOpReadVariableOp+policy_logits_add_1_readvariableop_resource*
_output_shapes
:*
dtype0�
policy_logits/add_1AddV2policy_logits/MatMul_1:output:0*policy_logits/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
!policy_logits/activation_239/ReluRelupolicy_logits/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
*flat_batch_norm_0/batchnorm/ReadVariableOpReadVariableOp3flat_batch_norm_0_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0f
!flat_batch_norm_0/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
flat_batch_norm_0/batchnorm/addAddV22flat_batch_norm_0/batchnorm/ReadVariableOp:value:0*flat_batch_norm_0/batchnorm/add/y:output:0*
T0*
_output_shapes
:t
!flat_batch_norm_0/batchnorm/RsqrtRsqrt#flat_batch_norm_0/batchnorm/add:z:0*
T0*
_output_shapes
:�
.flat_batch_norm_0/batchnorm/mul/ReadVariableOpReadVariableOp7flat_batch_norm_0_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
flat_batch_norm_0/batchnorm/mulMul%flat_batch_norm_0/batchnorm/Rsqrt:y:06flat_batch_norm_0/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
!flat_batch_norm_0/batchnorm/mul_1Mul"tf.math.reduce_sum_64/Sum:output:0#flat_batch_norm_0/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
,flat_batch_norm_0/batchnorm/ReadVariableOp_1ReadVariableOp5flat_batch_norm_0_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
!flat_batch_norm_0/batchnorm/mul_2Mul4flat_batch_norm_0/batchnorm/ReadVariableOp_1:value:0#flat_batch_norm_0/batchnorm/mul:z:0*
T0*
_output_shapes
:�
,flat_batch_norm_0/batchnorm/ReadVariableOp_2ReadVariableOp5flat_batch_norm_0_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
flat_batch_norm_0/batchnorm/subSub4flat_batch_norm_0/batchnorm/ReadVariableOp_2:value:0%flat_batch_norm_0/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
!flat_batch_norm_0/batchnorm/add_1AddV2%flat_batch_norm_0/batchnorm/mul_1:z:0#flat_batch_norm_0/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
0tf.__operators__.getitem_328/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_328/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_328/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_328/strided_sliceStridedSlice/policy_logits/activation_239/Relu:activations:09tf.__operators__.getitem_328/strided_slice/stack:output:0;tf.__operators__.getitem_328/strided_slice/stack_1:output:0;tf.__operators__.getitem_328/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_maska
tf.math.greater_59/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater_59/GreaterGreater3tf.__operators__.getitem_327/strided_slice:output:0%tf.math.greater_59/Greater/y:output:0*
T0*0
_output_shapes
:�������������������
flat_0/MatMul/ReadVariableOpReadVariableOp%flat_0_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
flat_0/MatMulMatMul%flat_batch_norm_0/batchnorm/add_1:z:0$flat_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
flat_0/BiasAdd/ReadVariableOpReadVariableOp&flat_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
flat_0/BiasAddBiasAddflat_0/MatMul:product:0%flat_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
flat_0/ReluReluflat_0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
policy_targets/ShapeShape3tf.__operators__.getitem_328/strided_slice:output:0*
T0*
_output_shapes
::��^
policy_targets/Fill/valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
policy_targets/FillFillpolicy_targets/Shape:output:0"policy_targets/Fill/value:output:0*
T0*0
_output_shapes
:�������������������
policy_targets/SelectV2SelectV2tf.math.greater_59/Greater:z:03tf.__operators__.getitem_328/strided_slice:output:0policy_targets/Fill:output:0*
T0*0
_output_shapes
:������������������~
policy_targets/SoftmaxSoftmax policy_targets/SelectV2:output:0*
T0*0
_output_shapes
:�������������������
#value_targets/MatMul/ReadVariableOpReadVariableOp,value_targets_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
value_targets/MatMulMatMulflat_0/Relu:activations:0+value_targets/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$value_targets/BiasAdd/ReadVariableOpReadVariableOp-value_targets_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
value_targets/BiasAddBiasAddvalue_targets/MatMul:product:0,value_targets/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
value_targets/TanhTanhvalue_targets/BiasAdd:output:0*
T0*'
_output_shapes
:���������e
IdentityIdentityvalue_targets/Tanh:y:0^NoOp*
T0*'
_output_shapes
:���������z

Identity_1Identity policy_targets/Softmax:softmax:0^NoOp*
T0*0
_output_shapes
:�������������������
NoOpNoOp^conv_0/MatMul_1/ReadVariableOp^conv_0/add_1/ReadVariableOp^conv_1/MatMul_1/ReadVariableOp^conv_1/add_1/ReadVariableOp+^conv_batch_norm_0/batchnorm/ReadVariableOp-^conv_batch_norm_0/batchnorm/ReadVariableOp_1-^conv_batch_norm_0/batchnorm/ReadVariableOp_2/^conv_batch_norm_0/batchnorm/mul/ReadVariableOp+^conv_batch_norm_1/batchnorm/ReadVariableOp-^conv_batch_norm_1/batchnorm/ReadVariableOp_1-^conv_batch_norm_1/batchnorm/ReadVariableOp_2/^conv_batch_norm_1/batchnorm/mul/ReadVariableOp^flat_0/BiasAdd/ReadVariableOp^flat_0/MatMul/ReadVariableOp+^flat_batch_norm_0/batchnorm/ReadVariableOp-^flat_batch_norm_0/batchnorm/ReadVariableOp_1-^flat_batch_norm_0/batchnorm/ReadVariableOp_2/^flat_batch_norm_0/batchnorm/mul/ReadVariableOp&^policy_logits/MatMul_1/ReadVariableOp#^policy_logits/add_1/ReadVariableOp%^value_targets/BiasAdd/ReadVariableOp$^value_targets/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 2@
conv_0/MatMul_1/ReadVariableOpconv_0/MatMul_1/ReadVariableOp2:
conv_0/add_1/ReadVariableOpconv_0/add_1/ReadVariableOp2@
conv_1/MatMul_1/ReadVariableOpconv_1/MatMul_1/ReadVariableOp2:
conv_1/add_1/ReadVariableOpconv_1/add_1/ReadVariableOp2\
,conv_batch_norm_0/batchnorm/ReadVariableOp_1,conv_batch_norm_0/batchnorm/ReadVariableOp_12\
,conv_batch_norm_0/batchnorm/ReadVariableOp_2,conv_batch_norm_0/batchnorm/ReadVariableOp_22X
*conv_batch_norm_0/batchnorm/ReadVariableOp*conv_batch_norm_0/batchnorm/ReadVariableOp2`
.conv_batch_norm_0/batchnorm/mul/ReadVariableOp.conv_batch_norm_0/batchnorm/mul/ReadVariableOp2\
,conv_batch_norm_1/batchnorm/ReadVariableOp_1,conv_batch_norm_1/batchnorm/ReadVariableOp_12\
,conv_batch_norm_1/batchnorm/ReadVariableOp_2,conv_batch_norm_1/batchnorm/ReadVariableOp_22X
*conv_batch_norm_1/batchnorm/ReadVariableOp*conv_batch_norm_1/batchnorm/ReadVariableOp2`
.conv_batch_norm_1/batchnorm/mul/ReadVariableOp.conv_batch_norm_1/batchnorm/mul/ReadVariableOp2>
flat_0/BiasAdd/ReadVariableOpflat_0/BiasAdd/ReadVariableOp2<
flat_0/MatMul/ReadVariableOpflat_0/MatMul/ReadVariableOp2\
,flat_batch_norm_0/batchnorm/ReadVariableOp_1,flat_batch_norm_0/batchnorm/ReadVariableOp_12\
,flat_batch_norm_0/batchnorm/ReadVariableOp_2,flat_batch_norm_0/batchnorm/ReadVariableOp_22X
*flat_batch_norm_0/batchnorm/ReadVariableOp*flat_batch_norm_0/batchnorm/ReadVariableOp2`
.flat_batch_norm_0/batchnorm/mul/ReadVariableOp.flat_batch_norm_0/batchnorm/mul/ReadVariableOp2N
%policy_logits/MatMul_1/ReadVariableOp%policy_logits/MatMul_1/ReadVariableOp2H
"policy_logits/add_1/ReadVariableOp"policy_logits/add_1/ReadVariableOp2L
$value_targets/BiasAdd/ReadVariableOp$value_targets/BiasAdd/ReadVariableOp2J
#value_targets/MatMul/ReadVariableOp#value_targets/MatMul/ReadVariableOp:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs_1:k g
A
_output_shapes/
-:+���������������������������
"
_user_specified_name
inputs_0
�
�
#__inference_signature_wrapper_84478
environment	
state
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallenvironmentstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:������������������:���������*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_83174x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:������������������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:JF
#
_output_shapes
:���������

_user_specified_namestate:n j
A
_output_shapes/
-:+���������������������������
%
_user_specified_nameenvironment
��
�
!__inference__traced_restore_86011
file_prefix0
assignvariableop_conv_0_kernel:,
assignvariableop_1_conv_0_bias:8
*assignvariableop_2_conv_batch_norm_0_gamma:7
)assignvariableop_3_conv_batch_norm_0_beta:>
0assignvariableop_4_conv_batch_norm_0_moving_mean:B
4assignvariableop_5_conv_batch_norm_0_moving_variance:2
 assignvariableop_6_conv_1_kernel:,
assignvariableop_7_conv_1_bias:8
*assignvariableop_8_conv_batch_norm_1_gamma:7
)assignvariableop_9_conv_batch_norm_1_beta:?
1assignvariableop_10_conv_batch_norm_1_moving_mean:C
5assignvariableop_11_conv_batch_norm_1_moving_variance:9
+assignvariableop_12_flat_batch_norm_0_gamma:8
*assignvariableop_13_flat_batch_norm_0_beta:?
1assignvariableop_14_flat_batch_norm_0_moving_mean:C
5assignvariableop_15_flat_batch_norm_0_moving_variance::
(assignvariableop_16_policy_logits_kernel:4
&assignvariableop_17_policy_logits_bias:4
!assignvariableop_18_flat_0_kernel:	�.
assignvariableop_19_flat_0_bias:	�;
(assignvariableop_20_value_targets_kernel:	�4
&assignvariableop_21_value_targets_bias:'
assignvariableop_22_iteration:	 +
!assignvariableop_23_learning_rate: %
assignvariableop_24_total_2: %
assignvariableop_25_count_2: %
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B1layer_with_weights-0/W/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/W/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/b/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-5/W/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-5/b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv_0_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_0_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp*assignvariableop_2_conv_batch_norm_0_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp)assignvariableop_3_conv_batch_norm_0_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp0assignvariableop_4_conv_batch_norm_0_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp4assignvariableop_5_conv_batch_norm_0_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp*assignvariableop_8_conv_batch_norm_1_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp)assignvariableop_9_conv_batch_norm_1_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp1assignvariableop_10_conv_batch_norm_1_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp5assignvariableop_11_conv_batch_norm_1_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_flat_batch_norm_0_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_flat_batch_norm_0_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_flat_batch_norm_0_moving_meanIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp5assignvariableop_15_flat_batch_norm_0_moving_varianceIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_policy_logits_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp&assignvariableop_17_policy_logits_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_flat_0_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_flat_0_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_value_targets_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp&assignvariableop_21_value_targets_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_iterationIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_learning_rateIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_2Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_2Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
&__inference_gnn_63_layer_call_fn_84062
environment	
state
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallenvironmentstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:���������:������������������*2
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gnn_63_layer_call_and_return_conditional_losses_84013o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������z

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*0
_output_shapes
:������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:JF
#
_output_shapes
:���������

_user_specified_namestate:n j
A
_output_shapes/
-:+���������������������������
%
_user_specified_nameenvironment
�m
�	
A__inference_gnn_63_layer_call_and_return_conditional_losses_83699
environment	
state
conv_0_83499:
conv_0_83501:%
conv_batch_norm_0_83504:%
conv_batch_norm_0_83506:%
conv_batch_norm_0_83508:%
conv_batch_norm_0_83510:
conv_1_83554:
conv_1_83556:%
conv_batch_norm_1_83559:%
conv_batch_norm_1_83561:%
conv_batch_norm_1_83563:%
conv_batch_norm_1_83565:%
policy_logits_83631:!
policy_logits_83633:%
flat_batch_norm_0_83636:%
flat_batch_norm_0_83638:%
flat_batch_norm_0_83640:%
flat_batch_norm_0_83642:
flat_0_83663:	�
flat_0_83665:	�&
value_targets_83692:	�!
value_targets_83694:
identity

identity_1��conv_0/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�)conv_batch_norm_0/StatefulPartitionedCall�)conv_batch_norm_1/StatefulPartitionedCall�flat_0/StatefulPartitionedCall�)flat_batch_norm_0/StatefulPartitionedCall�%policy_logits/StatefulPartitionedCall�%value_targets/StatefulPartitionedCalld
tf.compat.v1.shape_66/ShapeShapeenvironment*
T0*
_output_shapes
::��\
tf.cast_131/CastCaststate*

DstT0*

SrcT0*#
_output_shapes
:����������
reshape_65/PartitionedCallPartitionedCalltf.cast_131/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_65_layer_call_and_return_conditional_losses_83439z
0tf.__operators__.getitem_324/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*tf.__operators__.getitem_324/strided_sliceStridedSlice$tf.compat.v1.shape_66/Shape:output:09tf.__operators__.getitem_324/strided_slice/stack:output:0;tf.__operators__.getitem_324/strided_slice/stack_1:output:0;tf.__operators__.getitem_324/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
tf.one_hot_66/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
tf.one_hot_66/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
tf.one_hot_66/one_hotOneHot#reshape_65/PartitionedCall:output:03tf.__operators__.getitem_324/strided_slice:output:0'tf.one_hot_66/one_hot/on_value:output:0(tf.one_hot_66/one_hot/off_value:output:0*
TI0*
T0*4
_output_shapes"
 :�������������������
0tf.__operators__.getitem_325/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_325/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_325/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_325/strided_sliceStridedSliceenvironment9tf.__operators__.getitem_325/strided_slice/stack:output:0;tf.__operators__.getitem_325/strided_slice/stack_1:output:0;tf.__operators__.getitem_325/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask�
0tf.__operators__.getitem_326/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_326/strided_sliceStridedSliceenvironment9tf.__operators__.getitem_326/strided_slice/stack:output:0;tf.__operators__.getitem_326/strided_slice/stack_1:output:0;tf.__operators__.getitem_326/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask~
)tf.compat.v1.transpose_130/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_130/transpose	Transposetf.one_hot_66/one_hot:output:02tf.compat.v1.transpose_130/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
conv_0/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:0(tf.compat.v1.transpose_130/transpose:y:0conv_0_83499conv_0_83501*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_0_layer_call_and_return_conditional_losses_83498�
)conv_batch_norm_0/StatefulPartitionedCallStatefulPartitionedCall'conv_0/StatefulPartitionedCall:output:0conv_batch_norm_0_83504conv_batch_norm_0_83506conv_batch_norm_0_83508conv_batch_norm_0_83510*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_83209�
conv_1/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:02conv_batch_norm_0/StatefulPartitionedCall:output:0conv_1_83554conv_1_83556*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_83553�
)conv_batch_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_batch_norm_1_83559conv_batch_norm_1_83561conv_batch_norm_1_83563conv_batch_norm_1_83565*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_83291~
)tf.compat.v1.transpose_131/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_131/transpose	Transpose2conv_batch_norm_1/StatefulPartitionedCall:output:02tf.compat.v1.transpose_131/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
tf.math.multiply_65/MulMul(tf.compat.v1.transpose_131/transpose:y:0tf.one_hot_66/one_hot:output:0*
T0*4
_output_shapes"
 :�������������������
legals_mask/PartitionedCallPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:0tf.one_hot_66/one_hot:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_legals_mask_layer_call_and_return_conditional_losses_83582v
+tf.math.reduce_sum_64/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_64/SumSumtf.math.multiply_65/Mul:z:04tf.math.reduce_sum_64/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:����������
0tf.__operators__.getitem_327/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_327/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_327/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_327/strided_sliceStridedSlice$legals_mask/PartitionedCall:output:09tf.__operators__.getitem_327/strided_slice/stack:output:0;tf.__operators__.getitem_327/strided_slice/stack_1:output:0;tf.__operators__.getitem_327/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_mask�
%policy_logits/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:02conv_batch_norm_1/StatefulPartitionedCall:output:0policy_logits_83631policy_logits_83633*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_logits_layer_call_and_return_conditional_losses_83630�
)flat_batch_norm_0/StatefulPartitionedCallStatefulPartitionedCall"tf.math.reduce_sum_64/Sum:output:0flat_batch_norm_0_83636flat_batch_norm_0_83638flat_batch_norm_0_83640flat_batch_norm_0_83642*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_83373�
0tf.__operators__.getitem_328/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_328/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_328/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_328/strided_sliceStridedSlice.policy_logits/StatefulPartitionedCall:output:09tf.__operators__.getitem_328/strided_slice/stack:output:0;tf.__operators__.getitem_328/strided_slice/stack_1:output:0;tf.__operators__.getitem_328/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_maska
tf.math.greater_59/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater_59/GreaterGreater3tf.__operators__.getitem_327/strided_slice:output:0%tf.math.greater_59/Greater/y:output:0*
T0*0
_output_shapes
:�������������������
flat_0/StatefulPartitionedCallStatefulPartitionedCall2flat_batch_norm_0/StatefulPartitionedCall:output:0flat_0_83663flat_0_83665*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_flat_0_layer_call_and_return_conditional_losses_83662�
policy_targets/PartitionedCallPartitionedCall3tf.__operators__.getitem_328/strided_slice:output:0tf.math.greater_59/Greater:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_policy_targets_layer_call_and_return_conditional_losses_83678�
%value_targets/StatefulPartitionedCallStatefulPartitionedCall'flat_0/StatefulPartitionedCall:output:0value_targets_83692value_targets_83694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_value_targets_layer_call_and_return_conditional_losses_83691}
IdentityIdentity.value_targets/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������

Identity_1Identity'policy_targets/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�������������������
NoOpNoOp^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall*^conv_batch_norm_0/StatefulPartitionedCall*^conv_batch_norm_1/StatefulPartitionedCall^flat_0/StatefulPartitionedCall*^flat_batch_norm_0/StatefulPartitionedCall&^policy_logits/StatefulPartitionedCall&^value_targets/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2V
)conv_batch_norm_0/StatefulPartitionedCall)conv_batch_norm_0/StatefulPartitionedCall2V
)conv_batch_norm_1/StatefulPartitionedCall)conv_batch_norm_1/StatefulPartitionedCall2@
flat_0/StatefulPartitionedCallflat_0/StatefulPartitionedCall2V
)flat_batch_norm_0/StatefulPartitionedCall)flat_batch_norm_0/StatefulPartitionedCall2N
%policy_logits/StatefulPartitionedCall%policy_logits/StatefulPartitionedCall2N
%value_targets/StatefulPartitionedCall%value_targets/StatefulPartitionedCall:JF
#
_output_shapes
:���������

_user_specified_namestate:n j
A
_output_shapes/
-:+���������������������������
%
_user_specified_nameenvironment
�%
�
A__inference_conv_1_layer_call_and_return_conditional_losses_85371
inputs_0
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_238/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_238/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�
�
1__inference_flat_batch_norm_0_layer_call_fn_85494

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_83393o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
-__inference_policy_logits_layer_call_fn_85559
inputs_0
inputs_1
inputs_2
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_logits_layer_call_and_return_conditional_losses_83630|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�%
�
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_85431

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
W
+__inference_legals_mask_layer_call_fn_85457
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_legals_mask_layer_call_and_return_conditional_losses_83582m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:'���������������������������:������������������:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�
�
1__inference_conv_batch_norm_1_layer_call_fn_85384

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_83291|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
Z
.__inference_policy_targets_layer_call_fn_85696
inputs_0
inputs_1

identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2
*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_policy_targets_layer_call_and_return_conditional_losses_83678i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������:������������������:ZV
0
_output_shapes
:������������������
"
_user_specified_name
inputs_1:Z V
0
_output_shapes
:������������������
"
_user_specified_name
inputs_0
�%
�
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_83373

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_83311

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�

�
&__inference_conv_0_layer_call_fn_85109
inputs_0
inputs_1
inputs_2
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_0_layer_call_and_return_conditional_losses_83762|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
��
�
 __inference__wrapped_model_83174
environment	
state@
.gnn_63_conv_0_matmul_1_readvariableop_resource:9
+gnn_63_conv_0_add_1_readvariableop_resource:H
:gnn_63_conv_batch_norm_0_batchnorm_readvariableop_resource:L
>gnn_63_conv_batch_norm_0_batchnorm_mul_readvariableop_resource:J
<gnn_63_conv_batch_norm_0_batchnorm_readvariableop_1_resource:J
<gnn_63_conv_batch_norm_0_batchnorm_readvariableop_2_resource:@
.gnn_63_conv_1_matmul_1_readvariableop_resource:9
+gnn_63_conv_1_add_1_readvariableop_resource:H
:gnn_63_conv_batch_norm_1_batchnorm_readvariableop_resource:L
>gnn_63_conv_batch_norm_1_batchnorm_mul_readvariableop_resource:J
<gnn_63_conv_batch_norm_1_batchnorm_readvariableop_1_resource:J
<gnn_63_conv_batch_norm_1_batchnorm_readvariableop_2_resource:G
5gnn_63_policy_logits_matmul_1_readvariableop_resource:@
2gnn_63_policy_logits_add_1_readvariableop_resource:H
:gnn_63_flat_batch_norm_0_batchnorm_readvariableop_resource:L
>gnn_63_flat_batch_norm_0_batchnorm_mul_readvariableop_resource:J
<gnn_63_flat_batch_norm_0_batchnorm_readvariableop_1_resource:J
<gnn_63_flat_batch_norm_0_batchnorm_readvariableop_2_resource:?
,gnn_63_flat_0_matmul_readvariableop_resource:	�<
-gnn_63_flat_0_biasadd_readvariableop_resource:	�F
3gnn_63_value_targets_matmul_readvariableop_resource:	�B
4gnn_63_value_targets_biasadd_readvariableop_resource:
identity

identity_1��%gnn_63/conv_0/MatMul_1/ReadVariableOp�"gnn_63/conv_0/add_1/ReadVariableOp�%gnn_63/conv_1/MatMul_1/ReadVariableOp�"gnn_63/conv_1/add_1/ReadVariableOp�1gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp�3gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_1�3gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_2�5gnn_63/conv_batch_norm_0/batchnorm/mul/ReadVariableOp�1gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp�3gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_1�3gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_2�5gnn_63/conv_batch_norm_1/batchnorm/mul/ReadVariableOp�$gnn_63/flat_0/BiasAdd/ReadVariableOp�#gnn_63/flat_0/MatMul/ReadVariableOp�1gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp�3gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_1�3gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_2�5gnn_63/flat_batch_norm_0/batchnorm/mul/ReadVariableOp�,gnn_63/policy_logits/MatMul_1/ReadVariableOp�)gnn_63/policy_logits/add_1/ReadVariableOp�+gnn_63/value_targets/BiasAdd/ReadVariableOp�*gnn_63/value_targets/MatMul/ReadVariableOpk
"gnn_63/tf.compat.v1.shape_66/ShapeShapeenvironment*
T0*
_output_shapes
::��c
gnn_63/tf.cast_131/CastCaststate*

DstT0*

SrcT0*#
_output_shapes
:���������p
gnn_63/reshape_65/ShapeShapegnn_63/tf.cast_131/Cast:y:0*
T0*
_output_shapes
::��o
%gnn_63/reshape_65/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'gnn_63/reshape_65/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'gnn_63/reshape_65/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gnn_63/reshape_65/strided_sliceStridedSlice gnn_63/reshape_65/Shape:output:0.gnn_63/reshape_65/strided_slice/stack:output:00gnn_63/reshape_65/strided_slice/stack_1:output:00gnn_63/reshape_65/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!gnn_63/reshape_65/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
gnn_63/reshape_65/Reshape/shapePack(gnn_63/reshape_65/strided_slice:output:0*gnn_63/reshape_65/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
gnn_63/reshape_65/ReshapeReshapegnn_63/tf.cast_131/Cast:y:0(gnn_63/reshape_65/Reshape/shape:output:0*
T0*'
_output_shapes
:����������
7gnn_63/tf.__operators__.getitem_324/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:�
9gnn_63/tf.__operators__.getitem_324/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
9gnn_63/tf.__operators__.getitem_324/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
1gnn_63/tf.__operators__.getitem_324/strided_sliceStridedSlice+gnn_63/tf.compat.v1.shape_66/Shape:output:0@gnn_63/tf.__operators__.getitem_324/strided_slice/stack:output:0Bgnn_63/tf.__operators__.getitem_324/strided_slice/stack_1:output:0Bgnn_63/tf.__operators__.getitem_324/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
%gnn_63/tf.one_hot_66/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
&gnn_63/tf.one_hot_66/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gnn_63/tf.one_hot_66/one_hotOneHot"gnn_63/reshape_65/Reshape:output:0:gnn_63/tf.__operators__.getitem_324/strided_slice:output:0.gnn_63/tf.one_hot_66/one_hot/on_value:output:0/gnn_63/tf.one_hot_66/one_hot/off_value:output:0*
TI0*
T0*4
_output_shapes"
 :�������������������
7gnn_63/tf.__operators__.getitem_325/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
9gnn_63/tf.__operators__.getitem_325/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
9gnn_63/tf.__operators__.getitem_325/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
1gnn_63/tf.__operators__.getitem_325/strided_sliceStridedSliceenvironment@gnn_63/tf.__operators__.getitem_325/strided_slice/stack:output:0Bgnn_63/tf.__operators__.getitem_325/strided_slice/stack_1:output:0Bgnn_63/tf.__operators__.getitem_325/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask�
7gnn_63/tf.__operators__.getitem_326/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       �
9gnn_63/tf.__operators__.getitem_326/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
9gnn_63/tf.__operators__.getitem_326/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
1gnn_63/tf.__operators__.getitem_326/strided_sliceStridedSliceenvironment@gnn_63/tf.__operators__.getitem_326/strided_slice/stack:output:0Bgnn_63/tf.__operators__.getitem_326/strided_slice/stack_1:output:0Bgnn_63/tf.__operators__.getitem_326/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask�
0gnn_63/tf.compat.v1.transpose_130/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
+gnn_63/tf.compat.v1.transpose_130/transpose	Transpose%gnn_63/tf.one_hot_66/one_hot:output:09gnn_63/tf.compat.v1.transpose_130/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
gnn_63/conv_0/ShapeShape:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��t
!gnn_63/conv_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������m
#gnn_63/conv_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#gnn_63/conv_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gnn_63/conv_0/strided_sliceStridedSlicegnn_63/conv_0/Shape:output:0*gnn_63/conv_0/strided_slice/stack:output:0,gnn_63/conv_0/strided_slice/stack_1:output:0,gnn_63/conv_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
gnn_63/conv_0/Shape_1Shape:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��m
#gnn_63/conv_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%gnn_63/conv_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������o
%gnn_63/conv_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gnn_63/conv_0/strided_slice_1StridedSlicegnn_63/conv_0/Shape_1:output:0,gnn_63/conv_0/strided_slice_1/stack:output:0.gnn_63/conv_0/strided_slice_1/stack_1:output:0.gnn_63/conv_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
gnn_63/conv_0/eye/MinimumMinimum$gnn_63/conv_0/strided_slice:output:0$gnn_63/conv_0/strided_slice:output:0*
T0*
_output_shapes
: v
!gnn_63/conv_0/eye/concat/values_1Packgnn_63/conv_0/eye/Minimum:z:0*
N*
T0*
_output_shapes
:_
gnn_63/conv_0/eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
gnn_63/conv_0/eye/concatConcatV2&gnn_63/conv_0/strided_slice_1:output:0*gnn_63/conv_0/eye/concat/values_1:output:0&gnn_63/conv_0/eye/concat/axis:output:0*
N*
T0*
_output_shapes
:a
gnn_63/conv_0/eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gnn_63/conv_0/eye/onesFill!gnn_63/conv_0/eye/concat:output:0%gnn_63/conv_0/eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������Z
gnn_63/conv_0/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : j
gnn_63/conv_0/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������j
gnn_63/conv_0/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������i
$gnn_63/conv_0/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gnn_63/conv_0/eye/diagMatrixDiagV3gnn_63/conv_0/eye/ones:output:0!gnn_63/conv_0/eye/diag/k:output:0(gnn_63/conv_0/eye/diag/num_rows:output:0(gnn_63/conv_0/eye/diag/num_cols:output:0-gnn_63/conv_0/eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'����������������������������
gnn_63/conv_0/addAddV2:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0gnn_63/conv_0/eye/diag:output:0*
T0*=
_output_shapes+
):'����������������������������
gnn_63/conv_0/norm/mulMul:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0*
T0*=
_output_shapes+
):'���������������������������{
(gnn_63/conv_0/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
gnn_63/conv_0/norm/SumSumgnn_63/conv_0/norm/mul:z:01gnn_63/conv_0/norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(
gnn_63/conv_0/norm/SqrtSqrtgnn_63/conv_0/norm/Sum:output:0*
T0*4
_output_shapes"
 :�������������������
gnn_63/conv_0/MatMulBatchMatMulV2gnn_63/conv_0/add:z:0/gnn_63/tf.compat.v1.transpose_130/transpose:y:0*
T0*4
_output_shapes"
 :�������������������
%gnn_63/conv_0/MatMul_1/ReadVariableOpReadVariableOp.gnn_63_conv_0_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gnn_63/conv_0/MatMul_1BatchMatMulV2gnn_63/conv_0/MatMul:output:0-gnn_63/conv_0/MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
"gnn_63/conv_0/add_1/ReadVariableOpReadVariableOp+gnn_63_conv_0_add_1_readvariableop_resource*
_output_shapes
:*
dtype0�
gnn_63/conv_0/add_1AddV2gnn_63/conv_0/MatMul_1:output:0*gnn_63/conv_0/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
!gnn_63/conv_0/activation_237/ReluRelugnn_63/conv_0/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
1gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOpReadVariableOp:gnn_63_conv_batch_norm_0_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0m
(gnn_63/conv_batch_norm_0/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&gnn_63/conv_batch_norm_0/batchnorm/addAddV29gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp:value:01gnn_63/conv_batch_norm_0/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
(gnn_63/conv_batch_norm_0/batchnorm/RsqrtRsqrt*gnn_63/conv_batch_norm_0/batchnorm/add:z:0*
T0*
_output_shapes
:�
5gnn_63/conv_batch_norm_0/batchnorm/mul/ReadVariableOpReadVariableOp>gnn_63_conv_batch_norm_0_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
&gnn_63/conv_batch_norm_0/batchnorm/mulMul,gnn_63/conv_batch_norm_0/batchnorm/Rsqrt:y:0=gnn_63/conv_batch_norm_0/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(gnn_63/conv_batch_norm_0/batchnorm/mul_1Mul/gnn_63/conv_0/activation_237/Relu:activations:0*gnn_63/conv_batch_norm_0/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :�������������������
3gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_1ReadVariableOp<gnn_63_conv_batch_norm_0_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(gnn_63/conv_batch_norm_0/batchnorm/mul_2Mul;gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_1:value:0*gnn_63/conv_batch_norm_0/batchnorm/mul:z:0*
T0*
_output_shapes
:�
3gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_2ReadVariableOp<gnn_63_conv_batch_norm_0_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&gnn_63/conv_batch_norm_0/batchnorm/subSub;gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_2:value:0,gnn_63/conv_batch_norm_0/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(gnn_63/conv_batch_norm_0/batchnorm/add_1AddV2,gnn_63/conv_batch_norm_0/batchnorm/mul_1:z:0*gnn_63/conv_batch_norm_0/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :�������������������
gnn_63/conv_1/ShapeShape:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��t
!gnn_63/conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������m
#gnn_63/conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#gnn_63/conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gnn_63/conv_1/strided_sliceStridedSlicegnn_63/conv_1/Shape:output:0*gnn_63/conv_1/strided_slice/stack:output:0,gnn_63/conv_1/strided_slice/stack_1:output:0,gnn_63/conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
gnn_63/conv_1/Shape_1Shape:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��m
#gnn_63/conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%gnn_63/conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������o
%gnn_63/conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
gnn_63/conv_1/strided_slice_1StridedSlicegnn_63/conv_1/Shape_1:output:0,gnn_63/conv_1/strided_slice_1/stack:output:0.gnn_63/conv_1/strided_slice_1/stack_1:output:0.gnn_63/conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
gnn_63/conv_1/eye/MinimumMinimum$gnn_63/conv_1/strided_slice:output:0$gnn_63/conv_1/strided_slice:output:0*
T0*
_output_shapes
: v
!gnn_63/conv_1/eye/concat/values_1Packgnn_63/conv_1/eye/Minimum:z:0*
N*
T0*
_output_shapes
:_
gnn_63/conv_1/eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
gnn_63/conv_1/eye/concatConcatV2&gnn_63/conv_1/strided_slice_1:output:0*gnn_63/conv_1/eye/concat/values_1:output:0&gnn_63/conv_1/eye/concat/axis:output:0*
N*
T0*
_output_shapes
:a
gnn_63/conv_1/eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gnn_63/conv_1/eye/onesFill!gnn_63/conv_1/eye/concat:output:0%gnn_63/conv_1/eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������Z
gnn_63/conv_1/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : j
gnn_63/conv_1/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������j
gnn_63/conv_1/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������i
$gnn_63/conv_1/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gnn_63/conv_1/eye/diagMatrixDiagV3gnn_63/conv_1/eye/ones:output:0!gnn_63/conv_1/eye/diag/k:output:0(gnn_63/conv_1/eye/diag/num_rows:output:0(gnn_63/conv_1/eye/diag/num_cols:output:0-gnn_63/conv_1/eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'����������������������������
gnn_63/conv_1/addAddV2:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0gnn_63/conv_1/eye/diag:output:0*
T0*=
_output_shapes+
):'����������������������������
gnn_63/conv_1/norm/mulMul:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0*
T0*=
_output_shapes+
):'���������������������������{
(gnn_63/conv_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
gnn_63/conv_1/norm/SumSumgnn_63/conv_1/norm/mul:z:01gnn_63/conv_1/norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(
gnn_63/conv_1/norm/SqrtSqrtgnn_63/conv_1/norm/Sum:output:0*
T0*4
_output_shapes"
 :�������������������
gnn_63/conv_1/MatMulBatchMatMulV2gnn_63/conv_1/add:z:0,gnn_63/conv_batch_norm_0/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
%gnn_63/conv_1/MatMul_1/ReadVariableOpReadVariableOp.gnn_63_conv_1_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gnn_63/conv_1/MatMul_1BatchMatMulV2gnn_63/conv_1/MatMul:output:0-gnn_63/conv_1/MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
"gnn_63/conv_1/add_1/ReadVariableOpReadVariableOp+gnn_63_conv_1_add_1_readvariableop_resource*
_output_shapes
:*
dtype0�
gnn_63/conv_1/add_1AddV2gnn_63/conv_1/MatMul_1:output:0*gnn_63/conv_1/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
!gnn_63/conv_1/activation_238/ReluRelugnn_63/conv_1/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
1gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOpReadVariableOp:gnn_63_conv_batch_norm_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0m
(gnn_63/conv_batch_norm_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&gnn_63/conv_batch_norm_1/batchnorm/addAddV29gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp:value:01gnn_63/conv_batch_norm_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
(gnn_63/conv_batch_norm_1/batchnorm/RsqrtRsqrt*gnn_63/conv_batch_norm_1/batchnorm/add:z:0*
T0*
_output_shapes
:�
5gnn_63/conv_batch_norm_1/batchnorm/mul/ReadVariableOpReadVariableOp>gnn_63_conv_batch_norm_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
&gnn_63/conv_batch_norm_1/batchnorm/mulMul,gnn_63/conv_batch_norm_1/batchnorm/Rsqrt:y:0=gnn_63/conv_batch_norm_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(gnn_63/conv_batch_norm_1/batchnorm/mul_1Mul/gnn_63/conv_1/activation_238/Relu:activations:0*gnn_63/conv_batch_norm_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :�������������������
3gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_1ReadVariableOp<gnn_63_conv_batch_norm_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(gnn_63/conv_batch_norm_1/batchnorm/mul_2Mul;gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_1:value:0*gnn_63/conv_batch_norm_1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
3gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_2ReadVariableOp<gnn_63_conv_batch_norm_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&gnn_63/conv_batch_norm_1/batchnorm/subSub;gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_2:value:0,gnn_63/conv_batch_norm_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(gnn_63/conv_batch_norm_1/batchnorm/add_1AddV2,gnn_63/conv_batch_norm_1/batchnorm/mul_1:z:0*gnn_63/conv_batch_norm_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :�������������������
0gnn_63/tf.compat.v1.transpose_131/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
+gnn_63/tf.compat.v1.transpose_131/transpose	Transpose,gnn_63/conv_batch_norm_1/batchnorm/add_1:z:09gnn_63/tf.compat.v1.transpose_131/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
gnn_63/tf.math.multiply_65/MulMul/gnn_63/tf.compat.v1.transpose_131/transpose:y:0%gnn_63/tf.one_hot_66/one_hot:output:0*
T0*4
_output_shapes"
 :������������������v
!gnn_63/legals_mask/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gnn_63/legals_mask/transpose	Transpose:gnn_63/tf.__operators__.getitem_325/strided_slice:output:0*gnn_63/legals_mask/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������x
#gnn_63/legals_mask/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
gnn_63/legals_mask/transpose_1	Transpose%gnn_63/tf.one_hot_66/one_hot:output:0,gnn_63/legals_mask/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :�������������������
gnn_63/legals_mask/MatMulBatchMatMulV2 gnn_63/legals_mask/transpose:y:0"gnn_63/legals_mask/transpose_1:y:0*
T0*4
_output_shapes"
 :������������������x
gnn_63/legals_mask/ShapeShape"gnn_63/legals_mask/MatMul:output:0*
T0*
_output_shapes
::��}
2gnn_63/tf.math.reduce_sum_64/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
 gnn_63/tf.math.reduce_sum_64/SumSum"gnn_63/tf.math.multiply_65/Mul:z:0;gnn_63/tf.math.reduce_sum_64/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:����������
7gnn_63/tf.__operators__.getitem_327/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
9gnn_63/tf.__operators__.getitem_327/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
9gnn_63/tf.__operators__.getitem_327/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
1gnn_63/tf.__operators__.getitem_327/strided_sliceStridedSlice"gnn_63/legals_mask/MatMul:output:0@gnn_63/tf.__operators__.getitem_327/strided_slice/stack:output:0Bgnn_63/tf.__operators__.getitem_327/strided_slice/stack_1:output:0Bgnn_63/tf.__operators__.getitem_327/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_mask�
gnn_63/policy_logits/ShapeShape:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��{
(gnn_63/policy_logits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������t
*gnn_63/policy_logits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*gnn_63/policy_logits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"gnn_63/policy_logits/strided_sliceStridedSlice#gnn_63/policy_logits/Shape:output:01gnn_63/policy_logits/strided_slice/stack:output:03gnn_63/policy_logits/strided_slice/stack_1:output:03gnn_63/policy_logits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
gnn_63/policy_logits/Shape_1Shape:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��t
*gnn_63/policy_logits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
,gnn_63/policy_logits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������v
,gnn_63/policy_logits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$gnn_63/policy_logits/strided_slice_1StridedSlice%gnn_63/policy_logits/Shape_1:output:03gnn_63/policy_logits/strided_slice_1/stack:output:05gnn_63/policy_logits/strided_slice_1/stack_1:output:05gnn_63/policy_logits/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
 gnn_63/policy_logits/eye/MinimumMinimum+gnn_63/policy_logits/strided_slice:output:0+gnn_63/policy_logits/strided_slice:output:0*
T0*
_output_shapes
: �
(gnn_63/policy_logits/eye/concat/values_1Pack$gnn_63/policy_logits/eye/Minimum:z:0*
N*
T0*
_output_shapes
:f
$gnn_63/policy_logits/eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
gnn_63/policy_logits/eye/concatConcatV2-gnn_63/policy_logits/strided_slice_1:output:01gnn_63/policy_logits/eye/concat/values_1:output:0-gnn_63/policy_logits/eye/concat/axis:output:0*
N*
T0*
_output_shapes
:h
#gnn_63/policy_logits/eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gnn_63/policy_logits/eye/onesFill(gnn_63/policy_logits/eye/concat:output:0,gnn_63/policy_logits/eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������a
gnn_63/policy_logits/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : q
&gnn_63/policy_logits/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������q
&gnn_63/policy_logits/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������p
+gnn_63/policy_logits/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
gnn_63/policy_logits/eye/diagMatrixDiagV3&gnn_63/policy_logits/eye/ones:output:0(gnn_63/policy_logits/eye/diag/k:output:0/gnn_63/policy_logits/eye/diag/num_rows:output:0/gnn_63/policy_logits/eye/diag/num_cols:output:04gnn_63/policy_logits/eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'����������������������������
gnn_63/policy_logits/addAddV2:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0&gnn_63/policy_logits/eye/diag:output:0*
T0*=
_output_shapes+
):'����������������������������
gnn_63/policy_logits/norm/mulMul:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0:gnn_63/tf.__operators__.getitem_326/strided_slice:output:0*
T0*=
_output_shapes+
):'����������������������������
/gnn_63/policy_logits/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
gnn_63/policy_logits/norm/SumSum!gnn_63/policy_logits/norm/mul:z:08gnn_63/policy_logits/norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
gnn_63/policy_logits/norm/SqrtSqrt&gnn_63/policy_logits/norm/Sum:output:0*
T0*4
_output_shapes"
 :�������������������
gnn_63/policy_logits/MatMulBatchMatMulV2gnn_63/policy_logits/add:z:0,gnn_63/conv_batch_norm_1/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
,gnn_63/policy_logits/MatMul_1/ReadVariableOpReadVariableOp5gnn_63_policy_logits_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
gnn_63/policy_logits/MatMul_1BatchMatMulV2$gnn_63/policy_logits/MatMul:output:04gnn_63/policy_logits/MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
)gnn_63/policy_logits/add_1/ReadVariableOpReadVariableOp2gnn_63_policy_logits_add_1_readvariableop_resource*
_output_shapes
:*
dtype0�
gnn_63/policy_logits/add_1AddV2&gnn_63/policy_logits/MatMul_1:output:01gnn_63/policy_logits/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
(gnn_63/policy_logits/activation_239/ReluRelugnn_63/policy_logits/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
1gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOpReadVariableOp:gnn_63_flat_batch_norm_0_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0m
(gnn_63/flat_batch_norm_0/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
&gnn_63/flat_batch_norm_0/batchnorm/addAddV29gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp:value:01gnn_63/flat_batch_norm_0/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
(gnn_63/flat_batch_norm_0/batchnorm/RsqrtRsqrt*gnn_63/flat_batch_norm_0/batchnorm/add:z:0*
T0*
_output_shapes
:�
5gnn_63/flat_batch_norm_0/batchnorm/mul/ReadVariableOpReadVariableOp>gnn_63_flat_batch_norm_0_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
&gnn_63/flat_batch_norm_0/batchnorm/mulMul,gnn_63/flat_batch_norm_0/batchnorm/Rsqrt:y:0=gnn_63/flat_batch_norm_0/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
(gnn_63/flat_batch_norm_0/batchnorm/mul_1Mul)gnn_63/tf.math.reduce_sum_64/Sum:output:0*gnn_63/flat_batch_norm_0/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
3gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_1ReadVariableOp<gnn_63_flat_batch_norm_0_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
(gnn_63/flat_batch_norm_0/batchnorm/mul_2Mul;gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_1:value:0*gnn_63/flat_batch_norm_0/batchnorm/mul:z:0*
T0*
_output_shapes
:�
3gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_2ReadVariableOp<gnn_63_flat_batch_norm_0_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
&gnn_63/flat_batch_norm_0/batchnorm/subSub;gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_2:value:0,gnn_63/flat_batch_norm_0/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
(gnn_63/flat_batch_norm_0/batchnorm/add_1AddV2,gnn_63/flat_batch_norm_0/batchnorm/mul_1:z:0*gnn_63/flat_batch_norm_0/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
7gnn_63/tf.__operators__.getitem_328/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
9gnn_63/tf.__operators__.getitem_328/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
9gnn_63/tf.__operators__.getitem_328/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
1gnn_63/tf.__operators__.getitem_328/strided_sliceStridedSlice6gnn_63/policy_logits/activation_239/Relu:activations:0@gnn_63/tf.__operators__.getitem_328/strided_slice/stack:output:0Bgnn_63/tf.__operators__.getitem_328/strided_slice/stack_1:output:0Bgnn_63/tf.__operators__.getitem_328/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_maskh
#gnn_63/tf.math.greater_59/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
!gnn_63/tf.math.greater_59/GreaterGreater:gnn_63/tf.__operators__.getitem_327/strided_slice:output:0,gnn_63/tf.math.greater_59/Greater/y:output:0*
T0*0
_output_shapes
:�������������������
#gnn_63/flat_0/MatMul/ReadVariableOpReadVariableOp,gnn_63_flat_0_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gnn_63/flat_0/MatMulMatMul,gnn_63/flat_batch_norm_0/batchnorm/add_1:z:0+gnn_63/flat_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$gnn_63/flat_0/BiasAdd/ReadVariableOpReadVariableOp-gnn_63_flat_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gnn_63/flat_0/BiasAddBiasAddgnn_63/flat_0/MatMul:product:0,gnn_63/flat_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
gnn_63/flat_0/ReluRelugnn_63/flat_0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
gnn_63/policy_targets/ShapeShape:gnn_63/tf.__operators__.getitem_328/strided_slice:output:0*
T0*
_output_shapes
::��e
 gnn_63/policy_targets/Fill/valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
gnn_63/policy_targets/FillFill$gnn_63/policy_targets/Shape:output:0)gnn_63/policy_targets/Fill/value:output:0*
T0*0
_output_shapes
:�������������������
gnn_63/policy_targets/SelectV2SelectV2%gnn_63/tf.math.greater_59/Greater:z:0:gnn_63/tf.__operators__.getitem_328/strided_slice:output:0#gnn_63/policy_targets/Fill:output:0*
T0*0
_output_shapes
:�������������������
gnn_63/policy_targets/SoftmaxSoftmax'gnn_63/policy_targets/SelectV2:output:0*
T0*0
_output_shapes
:�������������������
*gnn_63/value_targets/MatMul/ReadVariableOpReadVariableOp3gnn_63_value_targets_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gnn_63/value_targets/MatMulMatMul gnn_63/flat_0/Relu:activations:02gnn_63/value_targets/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+gnn_63/value_targets/BiasAdd/ReadVariableOpReadVariableOp4gnn_63_value_targets_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
gnn_63/value_targets/BiasAddBiasAdd%gnn_63/value_targets/MatMul:product:03gnn_63/value_targets/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
gnn_63/value_targets/TanhTanh%gnn_63/value_targets/BiasAdd:output:0*
T0*'
_output_shapes
:���������
IdentityIdentity'gnn_63/policy_targets/Softmax:softmax:0^NoOp*
T0*0
_output_shapes
:������������������n

Identity_1Identitygnn_63/value_targets/Tanh:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^gnn_63/conv_0/MatMul_1/ReadVariableOp#^gnn_63/conv_0/add_1/ReadVariableOp&^gnn_63/conv_1/MatMul_1/ReadVariableOp#^gnn_63/conv_1/add_1/ReadVariableOp2^gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp4^gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_14^gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_26^gnn_63/conv_batch_norm_0/batchnorm/mul/ReadVariableOp2^gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp4^gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_14^gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_26^gnn_63/conv_batch_norm_1/batchnorm/mul/ReadVariableOp%^gnn_63/flat_0/BiasAdd/ReadVariableOp$^gnn_63/flat_0/MatMul/ReadVariableOp2^gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp4^gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_14^gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_26^gnn_63/flat_batch_norm_0/batchnorm/mul/ReadVariableOp-^gnn_63/policy_logits/MatMul_1/ReadVariableOp*^gnn_63/policy_logits/add_1/ReadVariableOp,^gnn_63/value_targets/BiasAdd/ReadVariableOp+^gnn_63/value_targets/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 2N
%gnn_63/conv_0/MatMul_1/ReadVariableOp%gnn_63/conv_0/MatMul_1/ReadVariableOp2H
"gnn_63/conv_0/add_1/ReadVariableOp"gnn_63/conv_0/add_1/ReadVariableOp2N
%gnn_63/conv_1/MatMul_1/ReadVariableOp%gnn_63/conv_1/MatMul_1/ReadVariableOp2H
"gnn_63/conv_1/add_1/ReadVariableOp"gnn_63/conv_1/add_1/ReadVariableOp2j
3gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_13gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_12j
3gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_23gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp_22f
1gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp1gnn_63/conv_batch_norm_0/batchnorm/ReadVariableOp2n
5gnn_63/conv_batch_norm_0/batchnorm/mul/ReadVariableOp5gnn_63/conv_batch_norm_0/batchnorm/mul/ReadVariableOp2j
3gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_13gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_12j
3gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_23gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp_22f
1gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp1gnn_63/conv_batch_norm_1/batchnorm/ReadVariableOp2n
5gnn_63/conv_batch_norm_1/batchnorm/mul/ReadVariableOp5gnn_63/conv_batch_norm_1/batchnorm/mul/ReadVariableOp2L
$gnn_63/flat_0/BiasAdd/ReadVariableOp$gnn_63/flat_0/BiasAdd/ReadVariableOp2J
#gnn_63/flat_0/MatMul/ReadVariableOp#gnn_63/flat_0/MatMul/ReadVariableOp2j
3gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_13gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_12j
3gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_23gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp_22f
1gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp1gnn_63/flat_batch_norm_0/batchnorm/ReadVariableOp2n
5gnn_63/flat_batch_norm_0/batchnorm/mul/ReadVariableOp5gnn_63/flat_batch_norm_0/batchnorm/mul/ReadVariableOp2\
,gnn_63/policy_logits/MatMul_1/ReadVariableOp,gnn_63/policy_logits/MatMul_1/ReadVariableOp2V
)gnn_63/policy_logits/add_1/ReadVariableOp)gnn_63/policy_logits/add_1/ReadVariableOp2Z
+gnn_63/value_targets/BiasAdd/ReadVariableOp+gnn_63/value_targets/BiasAdd/ReadVariableOp2X
*gnn_63/value_targets/MatMul/ReadVariableOp*gnn_63/value_targets/MatMul/ReadVariableOp:JF
#
_output_shapes
:���������

_user_specified_namestate:n j
A
_output_shapes/
-:+���������������������������
%
_user_specified_nameenvironment
�

�
A__inference_flat_0_layer_call_and_return_conditional_losses_83662

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_flat_0_layer_call_fn_85659

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_flat_0_layer_call_and_return_conditional_losses_83662p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
1__inference_conv_batch_norm_0_layer_call_fn_85215

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_83229|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�%
�
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_85249

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�	
r
F__inference_legals_mask_layer_call_and_return_conditional_losses_85468
inputs_0
inputs_1
identityc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
transpose_1	Transposeinputs_1transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������v
MatMulBatchMatMulV2transpose:y:0transpose_1:y:0*
T0*4
_output_shapes"
 :������������������R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��d
IdentityIdentityMatMul:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:'���������������������������:������������������:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�m
�	
A__inference_gnn_63_layer_call_and_return_conditional_losses_84013

inputs
inputs_1
conv_0_83942:
conv_0_83944:%
conv_batch_norm_0_83947:%
conv_batch_norm_0_83949:%
conv_batch_norm_0_83951:%
conv_batch_norm_0_83953:
conv_1_83956:
conv_1_83958:%
conv_batch_norm_1_83961:%
conv_batch_norm_1_83963:%
conv_batch_norm_1_83965:%
conv_batch_norm_1_83967:%
policy_logits_83980:!
policy_logits_83982:%
flat_batch_norm_0_83985:%
flat_batch_norm_0_83987:%
flat_batch_norm_0_83989:%
flat_batch_norm_0_83991:
flat_0_84000:	�
flat_0_84002:	�&
value_targets_84006:	�!
value_targets_84008:
identity

identity_1��conv_0/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�)conv_batch_norm_0/StatefulPartitionedCall�)conv_batch_norm_1/StatefulPartitionedCall�flat_0/StatefulPartitionedCall�)flat_batch_norm_0/StatefulPartitionedCall�%policy_logits/StatefulPartitionedCall�%value_targets/StatefulPartitionedCall_
tf.compat.v1.shape_66/ShapeShapeinputs*
T0*
_output_shapes
::��_
tf.cast_131/CastCastinputs_1*

DstT0*

SrcT0*#
_output_shapes
:����������
reshape_65/PartitionedCallPartitionedCalltf.cast_131/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_65_layer_call_and_return_conditional_losses_83439z
0tf.__operators__.getitem_324/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*tf.__operators__.getitem_324/strided_sliceStridedSlice$tf.compat.v1.shape_66/Shape:output:09tf.__operators__.getitem_324/strided_slice/stack:output:0;tf.__operators__.getitem_324/strided_slice/stack_1:output:0;tf.__operators__.getitem_324/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
tf.one_hot_66/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
tf.one_hot_66/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
tf.one_hot_66/one_hotOneHot#reshape_65/PartitionedCall:output:03tf.__operators__.getitem_324/strided_slice:output:0'tf.one_hot_66/one_hot/on_value:output:0(tf.one_hot_66/one_hot/off_value:output:0*
TI0*
T0*4
_output_shapes"
 :�������������������
0tf.__operators__.getitem_325/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_325/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_325/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_325/strided_sliceStridedSliceinputs9tf.__operators__.getitem_325/strided_slice/stack:output:0;tf.__operators__.getitem_325/strided_slice/stack_1:output:0;tf.__operators__.getitem_325/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask�
0tf.__operators__.getitem_326/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_326/strided_sliceStridedSliceinputs9tf.__operators__.getitem_326/strided_slice/stack:output:0;tf.__operators__.getitem_326/strided_slice/stack_1:output:0;tf.__operators__.getitem_326/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask~
)tf.compat.v1.transpose_130/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_130/transpose	Transposetf.one_hot_66/one_hot:output:02tf.compat.v1.transpose_130/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
conv_0/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:0(tf.compat.v1.transpose_130/transpose:y:0conv_0_83942conv_0_83944*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_0_layer_call_and_return_conditional_losses_83498�
)conv_batch_norm_0/StatefulPartitionedCallStatefulPartitionedCall'conv_0/StatefulPartitionedCall:output:0conv_batch_norm_0_83947conv_batch_norm_0_83949conv_batch_norm_0_83951conv_batch_norm_0_83953*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_83209�
conv_1/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:02conv_batch_norm_0/StatefulPartitionedCall:output:0conv_1_83956conv_1_83958*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_83553�
)conv_batch_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_batch_norm_1_83961conv_batch_norm_1_83963conv_batch_norm_1_83965conv_batch_norm_1_83967*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_83291~
)tf.compat.v1.transpose_131/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_131/transpose	Transpose2conv_batch_norm_1/StatefulPartitionedCall:output:02tf.compat.v1.transpose_131/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
tf.math.multiply_65/MulMul(tf.compat.v1.transpose_131/transpose:y:0tf.one_hot_66/one_hot:output:0*
T0*4
_output_shapes"
 :�������������������
legals_mask/PartitionedCallPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:0tf.one_hot_66/one_hot:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_legals_mask_layer_call_and_return_conditional_losses_83582v
+tf.math.reduce_sum_64/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_64/SumSumtf.math.multiply_65/Mul:z:04tf.math.reduce_sum_64/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:����������
0tf.__operators__.getitem_327/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_327/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_327/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_327/strided_sliceStridedSlice$legals_mask/PartitionedCall:output:09tf.__operators__.getitem_327/strided_slice/stack:output:0;tf.__operators__.getitem_327/strided_slice/stack_1:output:0;tf.__operators__.getitem_327/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_mask�
%policy_logits/StatefulPartitionedCallStatefulPartitionedCall3tf.__operators__.getitem_325/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:02conv_batch_norm_1/StatefulPartitionedCall:output:0policy_logits_83980policy_logits_83982*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_logits_layer_call_and_return_conditional_losses_83630�
)flat_batch_norm_0/StatefulPartitionedCallStatefulPartitionedCall"tf.math.reduce_sum_64/Sum:output:0flat_batch_norm_0_83985flat_batch_norm_0_83987flat_batch_norm_0_83989flat_batch_norm_0_83991*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_83373�
0tf.__operators__.getitem_328/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_328/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_328/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_328/strided_sliceStridedSlice.policy_logits/StatefulPartitionedCall:output:09tf.__operators__.getitem_328/strided_slice/stack:output:0;tf.__operators__.getitem_328/strided_slice/stack_1:output:0;tf.__operators__.getitem_328/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_maska
tf.math.greater_59/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater_59/GreaterGreater3tf.__operators__.getitem_327/strided_slice:output:0%tf.math.greater_59/Greater/y:output:0*
T0*0
_output_shapes
:�������������������
flat_0/StatefulPartitionedCallStatefulPartitionedCall2flat_batch_norm_0/StatefulPartitionedCall:output:0flat_0_84000flat_0_84002*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_flat_0_layer_call_and_return_conditional_losses_83662�
policy_targets/PartitionedCallPartitionedCall3tf.__operators__.getitem_328/strided_slice:output:0tf.math.greater_59/Greater:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_policy_targets_layer_call_and_return_conditional_losses_83678�
%value_targets/StatefulPartitionedCallStatefulPartitionedCall'flat_0/StatefulPartitionedCall:output:0value_targets_84006value_targets_84008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_value_targets_layer_call_and_return_conditional_losses_83691}
IdentityIdentity.value_targets/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������

Identity_1Identity'policy_targets/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�������������������
NoOpNoOp^conv_0/StatefulPartitionedCall^conv_1/StatefulPartitionedCall*^conv_batch_norm_0/StatefulPartitionedCall*^conv_batch_norm_1/StatefulPartitionedCall^flat_0/StatefulPartitionedCall*^flat_batch_norm_0/StatefulPartitionedCall&^policy_logits/StatefulPartitionedCall&^value_targets/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 2@
conv_0/StatefulPartitionedCallconv_0/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2V
)conv_batch_norm_0/StatefulPartitionedCall)conv_batch_norm_0/StatefulPartitionedCall2V
)conv_batch_norm_1/StatefulPartitionedCall)conv_batch_norm_1/StatefulPartitionedCall2@
flat_0/StatefulPartitionedCallflat_0/StatefulPartitionedCall2V
)flat_batch_norm_0/StatefulPartitionedCall)flat_batch_norm_0/StatefulPartitionedCall2N
%policy_logits/StatefulPartitionedCall%policy_logits/StatefulPartitionedCall2N
%value_targets/StatefulPartitionedCall%value_targets/StatefulPartitionedCall:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�%
�
H__inference_policy_logits_layer_call_and_return_conditional_losses_83630

inputs
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_239/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_239/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:\X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�	
a
E__inference_reshape_65_layer_call_and_return_conditional_losses_83439

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:���������:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
1__inference_conv_batch_norm_0_layer_call_fn_85202

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_83209|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�%
�
A__inference_conv_0_layer_call_and_return_conditional_losses_83498

inputs
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_237/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_237/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:\X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�

�
-__inference_policy_logits_layer_call_fn_85570
inputs_0
inputs_1
inputs_2
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_policy_logits_layer_call_and_return_conditional_losses_83880|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�%
�
A__inference_conv_0_layer_call_and_return_conditional_losses_85149
inputs_0
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_237/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_237/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�

�
H__inference_value_targets_layer_call_and_return_conditional_losses_85690

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
A__inference_conv_0_layer_call_and_return_conditional_losses_85189
inputs_0
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_237/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_237/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�%
�
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_83209

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
&__inference_gnn_63_layer_call_fn_84209
environment	
state
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallenvironmentstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:���������:������������������*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gnn_63_layer_call_and_return_conditional_losses_84160o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������z

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*0
_output_shapes
:������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:JF
#
_output_shapes
:���������

_user_specified_namestate:n j
A
_output_shapes/
-:+���������������������������
%
_user_specified_nameenvironment
�%
�
A__inference_conv_1_layer_call_and_return_conditional_losses_83816

inputs
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_238/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_238/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:\X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
��
�
__inference__traced_save_85911
file_prefix6
$read_disablecopyonread_conv_0_kernel:2
$read_1_disablecopyonread_conv_0_bias:>
0read_2_disablecopyonread_conv_batch_norm_0_gamma:=
/read_3_disablecopyonread_conv_batch_norm_0_beta:D
6read_4_disablecopyonread_conv_batch_norm_0_moving_mean:H
:read_5_disablecopyonread_conv_batch_norm_0_moving_variance:8
&read_6_disablecopyonread_conv_1_kernel:2
$read_7_disablecopyonread_conv_1_bias:>
0read_8_disablecopyonread_conv_batch_norm_1_gamma:=
/read_9_disablecopyonread_conv_batch_norm_1_beta:E
7read_10_disablecopyonread_conv_batch_norm_1_moving_mean:I
;read_11_disablecopyonread_conv_batch_norm_1_moving_variance:?
1read_12_disablecopyonread_flat_batch_norm_0_gamma:>
0read_13_disablecopyonread_flat_batch_norm_0_beta:E
7read_14_disablecopyonread_flat_batch_norm_0_moving_mean:I
;read_15_disablecopyonread_flat_batch_norm_0_moving_variance:@
.read_16_disablecopyonread_policy_logits_kernel::
,read_17_disablecopyonread_policy_logits_bias::
'read_18_disablecopyonread_flat_0_kernel:	�4
%read_19_disablecopyonread_flat_0_bias:	�A
.read_20_disablecopyonread_value_targets_kernel:	�:
,read_21_disablecopyonread_value_targets_bias:-
#read_22_disablecopyonread_iteration:	 1
'read_23_disablecopyonread_learning_rate: +
!read_24_disablecopyonread_total_2: +
!read_25_disablecopyonread_count_2: +
!read_26_disablecopyonread_total_1: +
!read_27_disablecopyonread_count_1: )
read_28_disablecopyonread_total: )
read_29_disablecopyonread_count: 
savev2_const
identity_61��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv_0_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv_0_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv_0_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv_0_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead0read_2_disablecopyonread_conv_batch_norm_0_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp0read_2_disablecopyonread_conv_batch_norm_0_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_3/DisableCopyOnReadDisableCopyOnRead/read_3_disablecopyonread_conv_batch_norm_0_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp/read_3_disablecopyonread_conv_batch_norm_0_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead6read_4_disablecopyonread_conv_batch_norm_0_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp6read_4_disablecopyonread_conv_batch_norm_0_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnRead:read_5_disablecopyonread_conv_batch_norm_0_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp:read_5_disablecopyonread_conv_batch_norm_0_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_conv_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_conv_1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:x
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_conv_1_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_conv_1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_8/DisableCopyOnReadDisableCopyOnRead0read_8_disablecopyonread_conv_batch_norm_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp0read_8_disablecopyonread_conv_batch_norm_1_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_9/DisableCopyOnReadDisableCopyOnRead/read_9_disablecopyonread_conv_batch_norm_1_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp/read_9_disablecopyonread_conv_batch_norm_1_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead7read_10_disablecopyonread_conv_batch_norm_1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp7read_10_disablecopyonread_conv_batch_norm_1_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_11/DisableCopyOnReadDisableCopyOnRead;read_11_disablecopyonread_conv_batch_norm_1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp;read_11_disablecopyonread_conv_batch_norm_1_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnRead1read_12_disablecopyonread_flat_batch_norm_0_gamma"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp1read_12_disablecopyonread_flat_batch_norm_0_gamma^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_13/DisableCopyOnReadDisableCopyOnRead0read_13_disablecopyonread_flat_batch_norm_0_beta"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp0read_13_disablecopyonread_flat_batch_norm_0_beta^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_14/DisableCopyOnReadDisableCopyOnRead7read_14_disablecopyonread_flat_batch_norm_0_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp7read_14_disablecopyonread_flat_batch_norm_0_moving_mean^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead;read_15_disablecopyonread_flat_batch_norm_0_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp;read_15_disablecopyonread_flat_batch_norm_0_moving_variance^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_policy_logits_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_policy_logits_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_17/DisableCopyOnReadDisableCopyOnRead,read_17_disablecopyonread_policy_logits_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp,read_17_disablecopyonread_policy_logits_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_18/DisableCopyOnReadDisableCopyOnRead'read_18_disablecopyonread_flat_0_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp'read_18_disablecopyonread_flat_0_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	�z
Read_19/DisableCopyOnReadDisableCopyOnRead%read_19_disablecopyonread_flat_0_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp%read_19_disablecopyonread_flat_0_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead.read_20_disablecopyonread_value_targets_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp.read_20_disablecopyonread_value_targets_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_21/DisableCopyOnReadDisableCopyOnRead,read_21_disablecopyonread_value_targets_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp,read_21_disablecopyonread_value_targets_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_22/DisableCopyOnReadDisableCopyOnRead#read_22_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp#read_22_disablecopyonread_iteration^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_learning_rate^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_24/DisableCopyOnReadDisableCopyOnRead!read_24_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp!read_24_disablecopyonread_total_2^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_25/DisableCopyOnReadDisableCopyOnRead!read_25_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp!read_25_disablecopyonread_count_2^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_26/DisableCopyOnReadDisableCopyOnRead!read_26_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp!read_26_disablecopyonread_total_1^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_27/DisableCopyOnReadDisableCopyOnRead!read_27_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp!read_27_disablecopyonread_count_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_28/DisableCopyOnReadDisableCopyOnReadread_28_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOpread_28_disablecopyonread_total^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_count^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B1layer_with_weights-0/W/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-0/b/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/W/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/b/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-5/W/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-5/b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_60Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_61IdentityIdentity_60:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_61Identity_61:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
A__inference_flat_0_layer_call_and_return_conditional_losses_85670

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
H__inference_policy_logits_layer_call_and_return_conditional_losses_85650
inputs_0
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_239/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_239/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�%
�
H__inference_policy_logits_layer_call_and_return_conditional_losses_85610
inputs_0
inputs_1
inputs_22
 matmul_1_readvariableop_resource:+
add_1_readvariableop_resource:
identity��MatMul_1/ReadVariableOp�add_1/ReadVariableOpK
ShapeShapeinputs_1*
T0*
_output_shapes
::��f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_1*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskg
eye/MinimumMinimumstrided_slice:output:0strided_slice:output:0*
T0*
_output_shapes
: Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2strided_slice_1:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?y
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������L

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'���������������������������q
addAddV2inputs_1eye/diag:output:0*
T0*=
_output_shapes+
):'���������������������������k
norm/mulMulinputs_1inputs_1*
T0*=
_output_shapes+
):'���������������������������m
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(c
	norm/SqrtSqrtnorm/Sum:output:0*
T0*4
_output_shapes"
 :������������������i
MatMulBatchMatMulV2add:z:0inputs_2*
T0*4
_output_shapes"
 :������������������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0~
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������e
activation_239/ReluRelu	add_1:z:0*
T0*4
_output_shapes"
 :������������������}
IdentityIdentity!activation_239/Relu:activations:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp^MatMul_1/ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
�
�
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_85451

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
&__inference_gnn_63_layer_call_fn_84530
inputs_0
inputs_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�

unknown_18:	�

unknown_19:	�

unknown_20:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:���������:������������������*2
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_gnn_63_layer_call_and_return_conditional_losses_84013o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������z

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*0
_output_shapes
:������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs_1:k g
A
_output_shapes/
-:+���������������������������
"
_user_specified_name
inputs_0
�

�
&__inference_conv_0_layer_call_fn_85098
inputs_0
inputs_1
inputs_2
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_conv_0_layer_call_and_return_conditional_losses_83498|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v:'���������������������������:'���������������������������:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_2:gc
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_1:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0
��
�
A__inference_gnn_63_layer_call_and_return_conditional_losses_84847
inputs_0
inputs_19
'conv_0_matmul_1_readvariableop_resource:2
$conv_0_add_1_readvariableop_resource:G
9conv_batch_norm_0_assignmovingavg_readvariableop_resource:I
;conv_batch_norm_0_assignmovingavg_1_readvariableop_resource:E
7conv_batch_norm_0_batchnorm_mul_readvariableop_resource:A
3conv_batch_norm_0_batchnorm_readvariableop_resource:9
'conv_1_matmul_1_readvariableop_resource:2
$conv_1_add_1_readvariableop_resource:G
9conv_batch_norm_1_assignmovingavg_readvariableop_resource:I
;conv_batch_norm_1_assignmovingavg_1_readvariableop_resource:E
7conv_batch_norm_1_batchnorm_mul_readvariableop_resource:A
3conv_batch_norm_1_batchnorm_readvariableop_resource:@
.policy_logits_matmul_1_readvariableop_resource:9
+policy_logits_add_1_readvariableop_resource:G
9flat_batch_norm_0_assignmovingavg_readvariableop_resource:I
;flat_batch_norm_0_assignmovingavg_1_readvariableop_resource:E
7flat_batch_norm_0_batchnorm_mul_readvariableop_resource:A
3flat_batch_norm_0_batchnorm_readvariableop_resource:8
%flat_0_matmul_readvariableop_resource:	�5
&flat_0_biasadd_readvariableop_resource:	�?
,value_targets_matmul_readvariableop_resource:	�;
-value_targets_biasadd_readvariableop_resource:
identity

identity_1��conv_0/MatMul_1/ReadVariableOp�conv_0/add_1/ReadVariableOp�conv_1/MatMul_1/ReadVariableOp�conv_1/add_1/ReadVariableOp�!conv_batch_norm_0/AssignMovingAvg�0conv_batch_norm_0/AssignMovingAvg/ReadVariableOp�#conv_batch_norm_0/AssignMovingAvg_1�2conv_batch_norm_0/AssignMovingAvg_1/ReadVariableOp�*conv_batch_norm_0/batchnorm/ReadVariableOp�.conv_batch_norm_0/batchnorm/mul/ReadVariableOp�!conv_batch_norm_1/AssignMovingAvg�0conv_batch_norm_1/AssignMovingAvg/ReadVariableOp�#conv_batch_norm_1/AssignMovingAvg_1�2conv_batch_norm_1/AssignMovingAvg_1/ReadVariableOp�*conv_batch_norm_1/batchnorm/ReadVariableOp�.conv_batch_norm_1/batchnorm/mul/ReadVariableOp�flat_0/BiasAdd/ReadVariableOp�flat_0/MatMul/ReadVariableOp�!flat_batch_norm_0/AssignMovingAvg�0flat_batch_norm_0/AssignMovingAvg/ReadVariableOp�#flat_batch_norm_0/AssignMovingAvg_1�2flat_batch_norm_0/AssignMovingAvg_1/ReadVariableOp�*flat_batch_norm_0/batchnorm/ReadVariableOp�.flat_batch_norm_0/batchnorm/mul/ReadVariableOp�%policy_logits/MatMul_1/ReadVariableOp�"policy_logits/add_1/ReadVariableOp�$value_targets/BiasAdd/ReadVariableOp�#value_targets/MatMul/ReadVariableOpa
tf.compat.v1.shape_66/ShapeShapeinputs_0*
T0*
_output_shapes
::��_
tf.cast_131/CastCastinputs_1*

DstT0*

SrcT0*#
_output_shapes
:���������b
reshape_65/ShapeShapetf.cast_131/Cast:y:0*
T0*
_output_shapes
::��h
reshape_65/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_65/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_65/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_65/strided_sliceStridedSlicereshape_65/Shape:output:0'reshape_65/strided_slice/stack:output:0)reshape_65/strided_slice/stack_1:output:0)reshape_65/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_65/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
reshape_65/Reshape/shapePack!reshape_65/strided_slice:output:0#reshape_65/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
reshape_65/ReshapeReshapetf.cast_131/Cast:y:0!reshape_65/Reshape/shape:output:0*
T0*'
_output_shapes
:���������z
0tf.__operators__.getitem_324/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2tf.__operators__.getitem_324/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*tf.__operators__.getitem_324/strided_sliceStridedSlice$tf.compat.v1.shape_66/Shape:output:09tf.__operators__.getitem_324/strided_slice/stack:output:0;tf.__operators__.getitem_324/strided_slice/stack_1:output:0;tf.__operators__.getitem_324/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
tf.one_hot_66/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
tf.one_hot_66/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
tf.one_hot_66/one_hotOneHotreshape_65/Reshape:output:03tf.__operators__.getitem_324/strided_slice:output:0'tf.one_hot_66/one_hot/on_value:output:0(tf.one_hot_66/one_hot/off_value:output:0*
TI0*
T0*4
_output_shapes"
 :�������������������
0tf.__operators__.getitem_325/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_325/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_325/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_325/strided_sliceStridedSliceinputs_09tf.__operators__.getitem_325/strided_slice/stack:output:0;tf.__operators__.getitem_325/strided_slice/stack_1:output:0;tf.__operators__.getitem_325/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask�
0tf.__operators__.getitem_326/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_326/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_326/strided_sliceStridedSliceinputs_09tf.__operators__.getitem_326/strided_slice/stack:output:0;tf.__operators__.getitem_326/strided_slice/stack_1:output:0;tf.__operators__.getitem_326/strided_slice/stack_2:output:0*
Index0*
T0*=
_output_shapes+
):'���������������������������*
ellipsis_mask*
shrink_axis_mask~
)tf.compat.v1.transpose_130/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_130/transpose	Transposetf.one_hot_66/one_hot:output:02tf.compat.v1.transpose_130/transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������}
conv_0/ShapeShape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��m
conv_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������f
conv_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
conv_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_0/strided_sliceStridedSliceconv_0/Shape:output:0#conv_0/strided_slice/stack:output:0%conv_0/strided_slice/stack_1:output:0%conv_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv_0/Shape_1Shape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��f
conv_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
conv_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������h
conv_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_0/strided_slice_1StridedSliceconv_0/Shape_1:output:0%conv_0/strided_slice_1/stack:output:0'conv_0/strided_slice_1/stack_1:output:0'conv_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
conv_0/eye/MinimumMinimumconv_0/strided_slice:output:0conv_0/strided_slice:output:0*
T0*
_output_shapes
: h
conv_0/eye/concat/values_1Packconv_0/eye/Minimum:z:0*
N*
T0*
_output_shapes
:X
conv_0/eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
conv_0/eye/concatConcatV2conv_0/strided_slice_1:output:0#conv_0/eye/concat/values_1:output:0conv_0/eye/concat/axis:output:0*
N*
T0*
_output_shapes
:Z
conv_0/eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv_0/eye/onesFillconv_0/eye/concat:output:0conv_0/eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������S
conv_0/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : c
conv_0/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������c
conv_0/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������b
conv_0/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_0/eye/diagMatrixDiagV3conv_0/eye/ones:output:0conv_0/eye/diag/k:output:0!conv_0/eye/diag/num_rows:output:0!conv_0/eye/diag/num_cols:output:0&conv_0/eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'����������������������������

conv_0/addAddV23tf.__operators__.getitem_326/strided_slice:output:0conv_0/eye/diag:output:0*
T0*=
_output_shapes+
):'����������������������������
conv_0/norm/mulMul3tf.__operators__.getitem_326/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:0*
T0*=
_output_shapes+
):'���������������������������t
!conv_0/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
conv_0/norm/SumSumconv_0/norm/mul:z:0*conv_0/norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(q
conv_0/norm/SqrtSqrtconv_0/norm/Sum:output:0*
T0*4
_output_shapes"
 :�������������������
conv_0/MatMulBatchMatMulV2conv_0/add:z:0(tf.compat.v1.transpose_130/transpose:y:0*
T0*4
_output_shapes"
 :�������������������
conv_0/MatMul_1/ReadVariableOpReadVariableOp'conv_0_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
conv_0/MatMul_1BatchMatMulV2conv_0/MatMul:output:0&conv_0/MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������|
conv_0/add_1/ReadVariableOpReadVariableOp$conv_0_add_1_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_0/add_1AddV2conv_0/MatMul_1:output:0#conv_0/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������s
conv_0/activation_237/ReluReluconv_0/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
0conv_batch_norm_0/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
conv_batch_norm_0/moments/meanMean(conv_0/activation_237/Relu:activations:09conv_batch_norm_0/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(�
&conv_batch_norm_0/moments/StopGradientStopGradient'conv_batch_norm_0/moments/mean:output:0*
T0*"
_output_shapes
:�
+conv_batch_norm_0/moments/SquaredDifferenceSquaredDifference(conv_0/activation_237/Relu:activations:0/conv_batch_norm_0/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :�������������������
4conv_batch_norm_0/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
"conv_batch_norm_0/moments/varianceMean/conv_batch_norm_0/moments/SquaredDifference:z:0=conv_batch_norm_0/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(�
!conv_batch_norm_0/moments/SqueezeSqueeze'conv_batch_norm_0/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
#conv_batch_norm_0/moments/Squeeze_1Squeeze+conv_batch_norm_0/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 l
'conv_batch_norm_0/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
0conv_batch_norm_0/AssignMovingAvg/ReadVariableOpReadVariableOp9conv_batch_norm_0_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
%conv_batch_norm_0/AssignMovingAvg/subSub8conv_batch_norm_0/AssignMovingAvg/ReadVariableOp:value:0*conv_batch_norm_0/moments/Squeeze:output:0*
T0*
_output_shapes
:�
%conv_batch_norm_0/AssignMovingAvg/mulMul)conv_batch_norm_0/AssignMovingAvg/sub:z:00conv_batch_norm_0/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
!conv_batch_norm_0/AssignMovingAvgAssignSubVariableOp9conv_batch_norm_0_assignmovingavg_readvariableop_resource)conv_batch_norm_0/AssignMovingAvg/mul:z:01^conv_batch_norm_0/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0n
)conv_batch_norm_0/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2conv_batch_norm_0/AssignMovingAvg_1/ReadVariableOpReadVariableOp;conv_batch_norm_0_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
'conv_batch_norm_0/AssignMovingAvg_1/subSub:conv_batch_norm_0/AssignMovingAvg_1/ReadVariableOp:value:0,conv_batch_norm_0/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
'conv_batch_norm_0/AssignMovingAvg_1/mulMul+conv_batch_norm_0/AssignMovingAvg_1/sub:z:02conv_batch_norm_0/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
#conv_batch_norm_0/AssignMovingAvg_1AssignSubVariableOp;conv_batch_norm_0_assignmovingavg_1_readvariableop_resource+conv_batch_norm_0/AssignMovingAvg_1/mul:z:03^conv_batch_norm_0/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0f
!conv_batch_norm_0/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv_batch_norm_0/batchnorm/addAddV2,conv_batch_norm_0/moments/Squeeze_1:output:0*conv_batch_norm_0/batchnorm/add/y:output:0*
T0*
_output_shapes
:t
!conv_batch_norm_0/batchnorm/RsqrtRsqrt#conv_batch_norm_0/batchnorm/add:z:0*
T0*
_output_shapes
:�
.conv_batch_norm_0/batchnorm/mul/ReadVariableOpReadVariableOp7conv_batch_norm_0_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_batch_norm_0/batchnorm/mulMul%conv_batch_norm_0/batchnorm/Rsqrt:y:06conv_batch_norm_0/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
!conv_batch_norm_0/batchnorm/mul_1Mul(conv_0/activation_237/Relu:activations:0#conv_batch_norm_0/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :�������������������
!conv_batch_norm_0/batchnorm/mul_2Mul*conv_batch_norm_0/moments/Squeeze:output:0#conv_batch_norm_0/batchnorm/mul:z:0*
T0*
_output_shapes
:�
*conv_batch_norm_0/batchnorm/ReadVariableOpReadVariableOp3conv_batch_norm_0_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_batch_norm_0/batchnorm/subSub2conv_batch_norm_0/batchnorm/ReadVariableOp:value:0%conv_batch_norm_0/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
!conv_batch_norm_0/batchnorm/add_1AddV2%conv_batch_norm_0/batchnorm/mul_1:z:0#conv_batch_norm_0/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������}
conv_1/ShapeShape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��m
conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������f
conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_1/strided_sliceStridedSliceconv_1/Shape:output:0#conv_1/strided_slice/stack:output:0%conv_1/strided_slice/stack_1:output:0%conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv_1/Shape_1Shape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��f
conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������h
conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv_1/strided_slice_1StridedSliceconv_1/Shape_1:output:0%conv_1/strided_slice_1/stack:output:0'conv_1/strided_slice_1/stack_1:output:0'conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
conv_1/eye/MinimumMinimumconv_1/strided_slice:output:0conv_1/strided_slice:output:0*
T0*
_output_shapes
: h
conv_1/eye/concat/values_1Packconv_1/eye/Minimum:z:0*
N*
T0*
_output_shapes
:X
conv_1/eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
conv_1/eye/concatConcatV2conv_1/strided_slice_1:output:0#conv_1/eye/concat/values_1:output:0conv_1/eye/concat/axis:output:0*
N*
T0*
_output_shapes
:Z
conv_1/eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv_1/eye/onesFillconv_1/eye/concat:output:0conv_1/eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������S
conv_1/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : c
conv_1/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������c
conv_1/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������b
conv_1/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv_1/eye/diagMatrixDiagV3conv_1/eye/ones:output:0conv_1/eye/diag/k:output:0!conv_1/eye/diag/num_rows:output:0!conv_1/eye/diag/num_cols:output:0&conv_1/eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'����������������������������

conv_1/addAddV23tf.__operators__.getitem_326/strided_slice:output:0conv_1/eye/diag:output:0*
T0*=
_output_shapes+
):'����������������������������
conv_1/norm/mulMul3tf.__operators__.getitem_326/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:0*
T0*=
_output_shapes+
):'���������������������������t
!conv_1/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
conv_1/norm/SumSumconv_1/norm/mul:z:0*conv_1/norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(q
conv_1/norm/SqrtSqrtconv_1/norm/Sum:output:0*
T0*4
_output_shapes"
 :�������������������
conv_1/MatMulBatchMatMulV2conv_1/add:z:0%conv_batch_norm_0/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
conv_1/MatMul_1/ReadVariableOpReadVariableOp'conv_1_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
conv_1/MatMul_1BatchMatMulV2conv_1/MatMul:output:0&conv_1/MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������|
conv_1/add_1/ReadVariableOpReadVariableOp$conv_1_add_1_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_1/add_1AddV2conv_1/MatMul_1:output:0#conv_1/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������s
conv_1/activation_238/ReluReluconv_1/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
0conv_batch_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
conv_batch_norm_1/moments/meanMean(conv_1/activation_238/Relu:activations:09conv_batch_norm_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(�
&conv_batch_norm_1/moments/StopGradientStopGradient'conv_batch_norm_1/moments/mean:output:0*
T0*"
_output_shapes
:�
+conv_batch_norm_1/moments/SquaredDifferenceSquaredDifference(conv_1/activation_238/Relu:activations:0/conv_batch_norm_1/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :�������������������
4conv_batch_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
"conv_batch_norm_1/moments/varianceMean/conv_batch_norm_1/moments/SquaredDifference:z:0=conv_batch_norm_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(�
!conv_batch_norm_1/moments/SqueezeSqueeze'conv_batch_norm_1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
#conv_batch_norm_1/moments/Squeeze_1Squeeze+conv_batch_norm_1/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 l
'conv_batch_norm_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
0conv_batch_norm_1/AssignMovingAvg/ReadVariableOpReadVariableOp9conv_batch_norm_1_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
%conv_batch_norm_1/AssignMovingAvg/subSub8conv_batch_norm_1/AssignMovingAvg/ReadVariableOp:value:0*conv_batch_norm_1/moments/Squeeze:output:0*
T0*
_output_shapes
:�
%conv_batch_norm_1/AssignMovingAvg/mulMul)conv_batch_norm_1/AssignMovingAvg/sub:z:00conv_batch_norm_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
!conv_batch_norm_1/AssignMovingAvgAssignSubVariableOp9conv_batch_norm_1_assignmovingavg_readvariableop_resource)conv_batch_norm_1/AssignMovingAvg/mul:z:01^conv_batch_norm_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0n
)conv_batch_norm_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2conv_batch_norm_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp;conv_batch_norm_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
'conv_batch_norm_1/AssignMovingAvg_1/subSub:conv_batch_norm_1/AssignMovingAvg_1/ReadVariableOp:value:0,conv_batch_norm_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
'conv_batch_norm_1/AssignMovingAvg_1/mulMul+conv_batch_norm_1/AssignMovingAvg_1/sub:z:02conv_batch_norm_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
#conv_batch_norm_1/AssignMovingAvg_1AssignSubVariableOp;conv_batch_norm_1_assignmovingavg_1_readvariableop_resource+conv_batch_norm_1/AssignMovingAvg_1/mul:z:03^conv_batch_norm_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0f
!conv_batch_norm_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
conv_batch_norm_1/batchnorm/addAddV2,conv_batch_norm_1/moments/Squeeze_1:output:0*conv_batch_norm_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:t
!conv_batch_norm_1/batchnorm/RsqrtRsqrt#conv_batch_norm_1/batchnorm/add:z:0*
T0*
_output_shapes
:�
.conv_batch_norm_1/batchnorm/mul/ReadVariableOpReadVariableOp7conv_batch_norm_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_batch_norm_1/batchnorm/mulMul%conv_batch_norm_1/batchnorm/Rsqrt:y:06conv_batch_norm_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
!conv_batch_norm_1/batchnorm/mul_1Mul(conv_1/activation_238/Relu:activations:0#conv_batch_norm_1/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :�������������������
!conv_batch_norm_1/batchnorm/mul_2Mul*conv_batch_norm_1/moments/Squeeze:output:0#conv_batch_norm_1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
*conv_batch_norm_1/batchnorm/ReadVariableOpReadVariableOp3conv_batch_norm_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
conv_batch_norm_1/batchnorm/subSub2conv_batch_norm_1/batchnorm/ReadVariableOp:value:0%conv_batch_norm_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
!conv_batch_norm_1/batchnorm/add_1AddV2%conv_batch_norm_1/batchnorm/mul_1:z:0#conv_batch_norm_1/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������~
)tf.compat.v1.transpose_131/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
$tf.compat.v1.transpose_131/transpose	Transpose%conv_batch_norm_1/batchnorm/add_1:z:02tf.compat.v1.transpose_131/transpose/perm:output:0*
T0*4
_output_shapes"
 :�������������������
tf.math.multiply_65/MulMul(tf.compat.v1.transpose_131/transpose:y:0tf.one_hot_66/one_hot:output:0*
T0*4
_output_shapes"
 :������������������o
legals_mask/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
legals_mask/transpose	Transpose3tf.__operators__.getitem_325/strided_slice:output:0#legals_mask/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������q
legals_mask/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
legals_mask/transpose_1	Transposetf.one_hot_66/one_hot:output:0%legals_mask/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :�������������������
legals_mask/MatMulBatchMatMulV2legals_mask/transpose:y:0legals_mask/transpose_1:y:0*
T0*4
_output_shapes"
 :������������������j
legals_mask/ShapeShapelegals_mask/MatMul:output:0*
T0*
_output_shapes
::��v
+tf.math.reduce_sum_64/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
tf.math.reduce_sum_64/SumSumtf.math.multiply_65/Mul:z:04tf.math.reduce_sum_64/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:����������
0tf.__operators__.getitem_327/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_327/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_327/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_327/strided_sliceStridedSlicelegals_mask/MatMul:output:09tf.__operators__.getitem_327/strided_slice/stack:output:0;tf.__operators__.getitem_327/strided_slice/stack_1:output:0;tf.__operators__.getitem_327/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_mask�
policy_logits/ShapeShape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��t
!policy_logits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������m
#policy_logits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#policy_logits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
policy_logits/strided_sliceStridedSlicepolicy_logits/Shape:output:0*policy_logits/strided_slice/stack:output:0,policy_logits/strided_slice/stack_1:output:0,policy_logits/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
policy_logits/Shape_1Shape3tf.__operators__.getitem_326/strided_slice:output:0*
T0*
_output_shapes
::��m
#policy_logits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%policy_logits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������o
%policy_logits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
policy_logits/strided_slice_1StridedSlicepolicy_logits/Shape_1:output:0,policy_logits/strided_slice_1/stack:output:0.policy_logits/strided_slice_1/stack_1:output:0.policy_logits/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
policy_logits/eye/MinimumMinimum$policy_logits/strided_slice:output:0$policy_logits/strided_slice:output:0*
T0*
_output_shapes
: v
!policy_logits/eye/concat/values_1Packpolicy_logits/eye/Minimum:z:0*
N*
T0*
_output_shapes
:_
policy_logits/eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
policy_logits/eye/concatConcatV2&policy_logits/strided_slice_1:output:0*policy_logits/eye/concat/values_1:output:0&policy_logits/eye/concat/axis:output:0*
N*
T0*
_output_shapes
:a
policy_logits/eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
policy_logits/eye/onesFill!policy_logits/eye/concat:output:0%policy_logits/eye/ones/Const:output:0*
T0*0
_output_shapes
:������������������Z
policy_logits/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : j
policy_logits/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������j
policy_logits/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������i
$policy_logits/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
policy_logits/eye/diagMatrixDiagV3policy_logits/eye/ones:output:0!policy_logits/eye/diag/k:output:0(policy_logits/eye/diag/num_rows:output:0(policy_logits/eye/diag/num_cols:output:0-policy_logits/eye/diag/padding_value:output:0*
T0*=
_output_shapes+
):'����������������������������
policy_logits/addAddV23tf.__operators__.getitem_326/strided_slice:output:0policy_logits/eye/diag:output:0*
T0*=
_output_shapes+
):'����������������������������
policy_logits/norm/mulMul3tf.__operators__.getitem_326/strided_slice:output:03tf.__operators__.getitem_326/strided_slice:output:0*
T0*=
_output_shapes+
):'���������������������������{
(policy_logits/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
policy_logits/norm/SumSumpolicy_logits/norm/mul:z:01policy_logits/norm/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(
policy_logits/norm/SqrtSqrtpolicy_logits/norm/Sum:output:0*
T0*4
_output_shapes"
 :�������������������
policy_logits/MatMulBatchMatMulV2policy_logits/add:z:0%conv_batch_norm_1/batchnorm/add_1:z:0*
T0*4
_output_shapes"
 :�������������������
%policy_logits/MatMul_1/ReadVariableOpReadVariableOp.policy_logits_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
policy_logits/MatMul_1BatchMatMulV2policy_logits/MatMul:output:0-policy_logits/MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
"policy_logits/add_1/ReadVariableOpReadVariableOp+policy_logits_add_1_readvariableop_resource*
_output_shapes
:*
dtype0�
policy_logits/add_1AddV2policy_logits/MatMul_1:output:0*policy_logits/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�������������������
!policy_logits/activation_239/ReluRelupolicy_logits/add_1:z:0*
T0*4
_output_shapes"
 :������������������z
0flat_batch_norm_0/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
flat_batch_norm_0/moments/meanMean"tf.math.reduce_sum_64/Sum:output:09flat_batch_norm_0/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
&flat_batch_norm_0/moments/StopGradientStopGradient'flat_batch_norm_0/moments/mean:output:0*
T0*
_output_shapes

:�
+flat_batch_norm_0/moments/SquaredDifferenceSquaredDifference"tf.math.reduce_sum_64/Sum:output:0/flat_batch_norm_0/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������~
4flat_batch_norm_0/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"flat_batch_norm_0/moments/varianceMean/flat_batch_norm_0/moments/SquaredDifference:z:0=flat_batch_norm_0/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
!flat_batch_norm_0/moments/SqueezeSqueeze'flat_batch_norm_0/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
#flat_batch_norm_0/moments/Squeeze_1Squeeze+flat_batch_norm_0/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 l
'flat_batch_norm_0/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
0flat_batch_norm_0/AssignMovingAvg/ReadVariableOpReadVariableOp9flat_batch_norm_0_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
%flat_batch_norm_0/AssignMovingAvg/subSub8flat_batch_norm_0/AssignMovingAvg/ReadVariableOp:value:0*flat_batch_norm_0/moments/Squeeze:output:0*
T0*
_output_shapes
:�
%flat_batch_norm_0/AssignMovingAvg/mulMul)flat_batch_norm_0/AssignMovingAvg/sub:z:00flat_batch_norm_0/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
!flat_batch_norm_0/AssignMovingAvgAssignSubVariableOp9flat_batch_norm_0_assignmovingavg_readvariableop_resource)flat_batch_norm_0/AssignMovingAvg/mul:z:01^flat_batch_norm_0/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0n
)flat_batch_norm_0/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2flat_batch_norm_0/AssignMovingAvg_1/ReadVariableOpReadVariableOp;flat_batch_norm_0_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
'flat_batch_norm_0/AssignMovingAvg_1/subSub:flat_batch_norm_0/AssignMovingAvg_1/ReadVariableOp:value:0,flat_batch_norm_0/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
'flat_batch_norm_0/AssignMovingAvg_1/mulMul+flat_batch_norm_0/AssignMovingAvg_1/sub:z:02flat_batch_norm_0/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
#flat_batch_norm_0/AssignMovingAvg_1AssignSubVariableOp;flat_batch_norm_0_assignmovingavg_1_readvariableop_resource+flat_batch_norm_0/AssignMovingAvg_1/mul:z:03^flat_batch_norm_0/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0f
!flat_batch_norm_0/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
flat_batch_norm_0/batchnorm/addAddV2,flat_batch_norm_0/moments/Squeeze_1:output:0*flat_batch_norm_0/batchnorm/add/y:output:0*
T0*
_output_shapes
:t
!flat_batch_norm_0/batchnorm/RsqrtRsqrt#flat_batch_norm_0/batchnorm/add:z:0*
T0*
_output_shapes
:�
.flat_batch_norm_0/batchnorm/mul/ReadVariableOpReadVariableOp7flat_batch_norm_0_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
flat_batch_norm_0/batchnorm/mulMul%flat_batch_norm_0/batchnorm/Rsqrt:y:06flat_batch_norm_0/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
!flat_batch_norm_0/batchnorm/mul_1Mul"tf.math.reduce_sum_64/Sum:output:0#flat_batch_norm_0/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
!flat_batch_norm_0/batchnorm/mul_2Mul*flat_batch_norm_0/moments/Squeeze:output:0#flat_batch_norm_0/batchnorm/mul:z:0*
T0*
_output_shapes
:�
*flat_batch_norm_0/batchnorm/ReadVariableOpReadVariableOp3flat_batch_norm_0_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
flat_batch_norm_0/batchnorm/subSub2flat_batch_norm_0/batchnorm/ReadVariableOp:value:0%flat_batch_norm_0/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
!flat_batch_norm_0/batchnorm/add_1AddV2%flat_batch_norm_0/batchnorm/mul_1:z:0#flat_batch_norm_0/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
0tf.__operators__.getitem_328/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
2tf.__operators__.getitem_328/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
2tf.__operators__.getitem_328/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
*tf.__operators__.getitem_328/strided_sliceStridedSlice/policy_logits/activation_239/Relu:activations:09tf.__operators__.getitem_328/strided_slice/stack:output:0;tf.__operators__.getitem_328/strided_slice/stack_1:output:0;tf.__operators__.getitem_328/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
ellipsis_mask*
shrink_axis_maska
tf.math.greater_59/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
tf.math.greater_59/GreaterGreater3tf.__operators__.getitem_327/strided_slice:output:0%tf.math.greater_59/Greater/y:output:0*
T0*0
_output_shapes
:�������������������
flat_0/MatMul/ReadVariableOpReadVariableOp%flat_0_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
flat_0/MatMulMatMul%flat_batch_norm_0/batchnorm/add_1:z:0$flat_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
flat_0/BiasAdd/ReadVariableOpReadVariableOp&flat_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
flat_0/BiasAddBiasAddflat_0/MatMul:product:0%flat_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������_
flat_0/ReluReluflat_0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
policy_targets/ShapeShape3tf.__operators__.getitem_328/strided_slice:output:0*
T0*
_output_shapes
::��^
policy_targets/Fill/valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
policy_targets/FillFillpolicy_targets/Shape:output:0"policy_targets/Fill/value:output:0*
T0*0
_output_shapes
:�������������������
policy_targets/SelectV2SelectV2tf.math.greater_59/Greater:z:03tf.__operators__.getitem_328/strided_slice:output:0policy_targets/Fill:output:0*
T0*0
_output_shapes
:������������������~
policy_targets/SoftmaxSoftmax policy_targets/SelectV2:output:0*
T0*0
_output_shapes
:�������������������
#value_targets/MatMul/ReadVariableOpReadVariableOp,value_targets_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
value_targets/MatMulMatMulflat_0/Relu:activations:0+value_targets/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$value_targets/BiasAdd/ReadVariableOpReadVariableOp-value_targets_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
value_targets/BiasAddBiasAddvalue_targets/MatMul:product:0,value_targets/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
value_targets/TanhTanhvalue_targets/BiasAdd:output:0*
T0*'
_output_shapes
:���������e
IdentityIdentityvalue_targets/Tanh:y:0^NoOp*
T0*'
_output_shapes
:���������z

Identity_1Identity policy_targets/Softmax:softmax:0^NoOp*
T0*0
_output_shapes
:�������������������	
NoOpNoOp^conv_0/MatMul_1/ReadVariableOp^conv_0/add_1/ReadVariableOp^conv_1/MatMul_1/ReadVariableOp^conv_1/add_1/ReadVariableOp"^conv_batch_norm_0/AssignMovingAvg1^conv_batch_norm_0/AssignMovingAvg/ReadVariableOp$^conv_batch_norm_0/AssignMovingAvg_13^conv_batch_norm_0/AssignMovingAvg_1/ReadVariableOp+^conv_batch_norm_0/batchnorm/ReadVariableOp/^conv_batch_norm_0/batchnorm/mul/ReadVariableOp"^conv_batch_norm_1/AssignMovingAvg1^conv_batch_norm_1/AssignMovingAvg/ReadVariableOp$^conv_batch_norm_1/AssignMovingAvg_13^conv_batch_norm_1/AssignMovingAvg_1/ReadVariableOp+^conv_batch_norm_1/batchnorm/ReadVariableOp/^conv_batch_norm_1/batchnorm/mul/ReadVariableOp^flat_0/BiasAdd/ReadVariableOp^flat_0/MatMul/ReadVariableOp"^flat_batch_norm_0/AssignMovingAvg1^flat_batch_norm_0/AssignMovingAvg/ReadVariableOp$^flat_batch_norm_0/AssignMovingAvg_13^flat_batch_norm_0/AssignMovingAvg_1/ReadVariableOp+^flat_batch_norm_0/batchnorm/ReadVariableOp/^flat_batch_norm_0/batchnorm/mul/ReadVariableOp&^policy_logits/MatMul_1/ReadVariableOp#^policy_logits/add_1/ReadVariableOp%^value_targets/BiasAdd/ReadVariableOp$^value_targets/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:+���������������������������:���������: : : : : : : : : : : : : : : : : : : : : : 2@
conv_0/MatMul_1/ReadVariableOpconv_0/MatMul_1/ReadVariableOp2:
conv_0/add_1/ReadVariableOpconv_0/add_1/ReadVariableOp2@
conv_1/MatMul_1/ReadVariableOpconv_1/MatMul_1/ReadVariableOp2:
conv_1/add_1/ReadVariableOpconv_1/add_1/ReadVariableOp2d
0conv_batch_norm_0/AssignMovingAvg/ReadVariableOp0conv_batch_norm_0/AssignMovingAvg/ReadVariableOp2h
2conv_batch_norm_0/AssignMovingAvg_1/ReadVariableOp2conv_batch_norm_0/AssignMovingAvg_1/ReadVariableOp2J
#conv_batch_norm_0/AssignMovingAvg_1#conv_batch_norm_0/AssignMovingAvg_12F
!conv_batch_norm_0/AssignMovingAvg!conv_batch_norm_0/AssignMovingAvg2X
*conv_batch_norm_0/batchnorm/ReadVariableOp*conv_batch_norm_0/batchnorm/ReadVariableOp2`
.conv_batch_norm_0/batchnorm/mul/ReadVariableOp.conv_batch_norm_0/batchnorm/mul/ReadVariableOp2d
0conv_batch_norm_1/AssignMovingAvg/ReadVariableOp0conv_batch_norm_1/AssignMovingAvg/ReadVariableOp2h
2conv_batch_norm_1/AssignMovingAvg_1/ReadVariableOp2conv_batch_norm_1/AssignMovingAvg_1/ReadVariableOp2J
#conv_batch_norm_1/AssignMovingAvg_1#conv_batch_norm_1/AssignMovingAvg_12F
!conv_batch_norm_1/AssignMovingAvg!conv_batch_norm_1/AssignMovingAvg2X
*conv_batch_norm_1/batchnorm/ReadVariableOp*conv_batch_norm_1/batchnorm/ReadVariableOp2`
.conv_batch_norm_1/batchnorm/mul/ReadVariableOp.conv_batch_norm_1/batchnorm/mul/ReadVariableOp2>
flat_0/BiasAdd/ReadVariableOpflat_0/BiasAdd/ReadVariableOp2<
flat_0/MatMul/ReadVariableOpflat_0/MatMul/ReadVariableOp2d
0flat_batch_norm_0/AssignMovingAvg/ReadVariableOp0flat_batch_norm_0/AssignMovingAvg/ReadVariableOp2h
2flat_batch_norm_0/AssignMovingAvg_1/ReadVariableOp2flat_batch_norm_0/AssignMovingAvg_1/ReadVariableOp2J
#flat_batch_norm_0/AssignMovingAvg_1#flat_batch_norm_0/AssignMovingAvg_12F
!flat_batch_norm_0/AssignMovingAvg!flat_batch_norm_0/AssignMovingAvg2X
*flat_batch_norm_0/batchnorm/ReadVariableOp*flat_batch_norm_0/batchnorm/ReadVariableOp2`
.flat_batch_norm_0/batchnorm/mul/ReadVariableOp.flat_batch_norm_0/batchnorm/mul/ReadVariableOp2N
%policy_logits/MatMul_1/ReadVariableOp%policy_logits/MatMul_1/ReadVariableOp2H
"policy_logits/add_1/ReadVariableOp"policy_logits/add_1/ReadVariableOp2L
$value_targets/BiasAdd/ReadVariableOp$value_targets/BiasAdd/ReadVariableOp2J
#value_targets/MatMul/ReadVariableOp#value_targets/MatMul/ReadVariableOp:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs_1:k g
A
_output_shapes/
-:+���������������������������
"
_user_specified_name
inputs_0"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
]
environmentN
serving_default_environment:0+���������������������������
3
state*
serving_default_state:0���������K
policy_targets9
StatefulPartitionedCall:0������������������A
value_targets0
StatefulPartitionedCall:1���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-0
layer-11
layer_with_weights-1
layer-12
layer_with_weights-2
layer-13
layer_with_weights-3
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-4
layer-19
layer_with_weights-5
layer-20
layer-21
layer_with_weights-6
layer-22
layer-23
layer-24
layer_with_weights-7
layer-25
layer-26
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_default_save_signature
#	optimizer
$loss
%
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
(
&	keras_api"
_tf_keras_layer
(
'	keras_api"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
(
.	keras_api"
_tf_keras_layer
(
/	keras_api"
_tf_keras_layer
(
0	keras_api"
_tf_keras_layer
(
1	keras_api"
_tf_keras_layer
(
2	keras_api"
_tf_keras_layer
(
3	keras_api"
_tf_keras_layer
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:
activation
;aggregation
<W
=b"
_tf_keras_layer
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
Daxis
	Egamma
Fbeta
Gmoving_mean
Hmoving_variance"
_tf_keras_layer
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O
activation
Paggregation
QW
Rb"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance"
_tf_keras_layer
(
^	keras_api"
_tf_keras_layer
(
_	keras_api"
_tf_keras_layer
(
`	keras_api"
_tf_keras_layer
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance"
_tf_keras_layer
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
x
activation
yaggregation
zW
{b"
_tf_keras_layer
(
|	keras_api"
_tf_keras_layer
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
<0
=1
E2
F3
G4
H5
Q6
R7
Z8
[9
\10
]11
n12
o13
p14
q15
z16
{17
�18
�19
�20
�21"
trackable_list_wrapper
�
<0
=1
E2
F3
Q4
R5
Z6
[7
n8
o9
z10
{11
�12
�13
�14
�15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
&__inference_gnn_63_layer_call_fn_84062
&__inference_gnn_63_layer_call_fn_84209
&__inference_gnn_63_layer_call_fn_84530
&__inference_gnn_63_layer_call_fn_84582�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
A__inference_gnn_63_layer_call_and_return_conditional_losses_83699
A__inference_gnn_63_layer_call_and_return_conditional_losses_83914
A__inference_gnn_63_layer_call_and_return_conditional_losses_84847
A__inference_gnn_63_layer_call_and_return_conditional_losses_85070�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_83174environmentstate"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
n
�
_variables
�_iterations
�_learning_rate
�_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
-
�serving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_reshape_65_layer_call_fn_85075�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_reshape_65_layer_call_and_return_conditional_losses_85087�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
&__inference_conv_0_layer_call_fn_85098
&__inference_conv_0_layer_call_fn_85109�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
A__inference_conv_0_layer_call_and_return_conditional_losses_85149
A__inference_conv_0_layer_call_and_return_conditional_losses_85189�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_generic_user_object
:2conv_0/kernel
:2conv_0/bias
<
E0
F1
G2
H3"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_conv_batch_norm_0_layer_call_fn_85202
1__inference_conv_batch_norm_0_layer_call_fn_85215�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_85249
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_85269�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
%:#2conv_batch_norm_0/gamma
$:"2conv_batch_norm_0/beta
-:+ (2conv_batch_norm_0/moving_mean
1:/ (2!conv_batch_norm_0/moving_variance
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
&__inference_conv_1_layer_call_fn_85280
&__inference_conv_1_layer_call_fn_85291�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
A__inference_conv_1_layer_call_and_return_conditional_losses_85331
A__inference_conv_1_layer_call_and_return_conditional_losses_85371�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_generic_user_object
:2conv_1/kernel
:2conv_1/bias
<
Z0
[1
\2
]3"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_conv_batch_norm_1_layer_call_fn_85384
1__inference_conv_batch_norm_1_layer_call_fn_85397�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_85431
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_85451�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
%:#2conv_batch_norm_1/gamma
$:"2conv_batch_norm_1/beta
-:+ (2conv_batch_norm_1/moving_mean
1:/ (2!conv_batch_norm_1/moving_variance
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_legals_mask_layer_call_fn_85457�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_legals_mask_layer_call_and_return_conditional_losses_85468�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
n0
o1
p2
q3"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_flat_batch_norm_0_layer_call_fn_85481
1__inference_flat_batch_norm_0_layer_call_fn_85494�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_85528
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_85548�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
%:#2flat_batch_norm_0/gamma
$:"2flat_batch_norm_0/beta
-:+ (2flat_batch_norm_0/moving_mean
1:/ (2!flat_batch_norm_0/moving_variance
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_policy_logits_layer_call_fn_85559
-__inference_policy_logits_layer_call_fn_85570�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_policy_logits_layer_call_and_return_conditional_losses_85610
H__inference_policy_logits_layer_call_and_return_conditional_losses_85650�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_generic_user_object
&:$2policy_logits/kernel
 :2policy_logits/bias
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_flat_0_layer_call_fn_85659�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_flat_0_layer_call_and_return_conditional_losses_85670�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 :	�2flat_0/kernel
:�2flat_0/bias
"
_generic_user_object
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_value_targets_layer_call_fn_85679�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_value_targets_layer_call_and_return_conditional_losses_85690�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
':%	�2value_targets/kernel
 :2value_targets/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_policy_targets_layer_call_fn_85696�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_policy_targets_layer_call_and_return_conditional_losses_85706�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
J
G0
H1
\2
]3
p4
q5"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_gnn_63_layer_call_fn_84062environmentstate"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_gnn_63_layer_call_fn_84209environmentstate"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_gnn_63_layer_call_fn_84530inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_gnn_63_layer_call_fn_84582inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_gnn_63_layer_call_and_return_conditional_losses_83699environmentstate"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_gnn_63_layer_call_and_return_conditional_losses_83914environmentstate"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_gnn_63_layer_call_and_return_conditional_losses_84847inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_gnn_63_layer_call_and_return_conditional_losses_85070inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
#__inference_signature_wrapper_84478environmentstate"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_reshape_65_layer_call_fn_85075inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_reshape_65_layer_call_and_return_conditional_losses_85087inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv_0_layer_call_fn_85098inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
&__inference_conv_0_layer_call_fn_85109inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
A__inference_conv_0_layer_call_and_return_conditional_losses_85149inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
A__inference_conv_0_layer_call_and_return_conditional_losses_85189inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_conv_batch_norm_0_layer_call_fn_85202inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_conv_batch_norm_0_layer_call_fn_85215inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_85249inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_85269inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
O0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv_1_layer_call_fn_85280inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
&__inference_conv_1_layer_call_fn_85291inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
A__inference_conv_1_layer_call_and_return_conditional_losses_85331inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
A__inference_conv_1_layer_call_and_return_conditional_losses_85371inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_conv_batch_norm_1_layer_call_fn_85384inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_conv_batch_norm_1_layer_call_fn_85397inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_85431inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_85451inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_legals_mask_layer_call_fn_85457inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_legals_mask_layer_call_and_return_conditional_losses_85468inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_flat_batch_norm_0_layer_call_fn_85481inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_flat_batch_norm_0_layer_call_fn_85494inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_85528inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_85548inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
x0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_policy_logits_layer_call_fn_85559inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
-__inference_policy_logits_layer_call_fn_85570inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
H__inference_policy_logits_layer_call_and_return_conditional_losses_85610inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
H__inference_policy_logits_layer_call_and_return_conditional_losses_85650inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_flat_0_layer_call_fn_85659inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_flat_0_layer_call_and_return_conditional_losses_85670inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_value_targets_layer_call_fn_85679inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_value_targets_layer_call_and_return_conditional_losses_85690inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_policy_targets_layer_call_fn_85696inputs_0inputs_1"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_policy_targets_layer_call_and_return_conditional_losses_85706inputs_0inputs_1"�
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
 __inference__wrapped_model_83174�<=HEGFQR]Z\[z{qnpo����p�m
f�c
a�^
?�<
environment+���������������������������
�
state���������
� "��
C
policy_targets1�.
policy_targets������������������
8
value_targets'�$
value_targets����������
A__inference_conv_0_layer_call_and_return_conditional_losses_85149�<=���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp"9�6
/�,
tensor_0������������������
� �
A__inference_conv_0_layer_call_and_return_conditional_losses_85189�<=���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp "9�6
/�,
tensor_0������������������
� �
&__inference_conv_0_layer_call_fn_85098�<=���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp".�+
unknown�������������������
&__inference_conv_0_layer_call_fn_85109�<=���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp ".�+
unknown�������������������
A__inference_conv_1_layer_call_and_return_conditional_losses_85331�QR���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp"9�6
/�,
tensor_0������������������
� �
A__inference_conv_1_layer_call_and_return_conditional_losses_85371�QR���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp "9�6
/�,
tensor_0������������������
� �
&__inference_conv_1_layer_call_fn_85280�QR���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp".�+
unknown�������������������
&__inference_conv_1_layer_call_fn_85291�QR���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp ".�+
unknown�������������������
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_85249�GHEFD�A
:�7
-�*
inputs������������������
p

 
� "9�6
/�,
tensor_0������������������
� �
L__inference_conv_batch_norm_0_layer_call_and_return_conditional_losses_85269�HEGFD�A
:�7
-�*
inputs������������������
p 

 
� "9�6
/�,
tensor_0������������������
� �
1__inference_conv_batch_norm_0_layer_call_fn_85202|GHEFD�A
:�7
-�*
inputs������������������
p

 
� ".�+
unknown�������������������
1__inference_conv_batch_norm_0_layer_call_fn_85215|HEGFD�A
:�7
-�*
inputs������������������
p 

 
� ".�+
unknown�������������������
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_85431�\]Z[D�A
:�7
-�*
inputs������������������
p

 
� "9�6
/�,
tensor_0������������������
� �
L__inference_conv_batch_norm_1_layer_call_and_return_conditional_losses_85451�]Z\[D�A
:�7
-�*
inputs������������������
p 

 
� "9�6
/�,
tensor_0������������������
� �
1__inference_conv_batch_norm_1_layer_call_fn_85384|\]Z[D�A
:�7
-�*
inputs������������������
p

 
� ".�+
unknown�������������������
1__inference_conv_batch_norm_1_layer_call_fn_85397|]Z\[D�A
:�7
-�*
inputs������������������
p 

 
� ".�+
unknown�������������������
A__inference_flat_0_layer_call_and_return_conditional_losses_85670f��/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
&__inference_flat_0_layer_call_fn_85659[��/�,
%�"
 �
inputs���������
� ""�
unknown�����������
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_85528mpqno7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
L__inference_flat_batch_norm_0_layer_call_and_return_conditional_losses_85548mqnpo7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
1__inference_flat_batch_norm_0_layer_call_fn_85481bpqno7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
1__inference_flat_batch_norm_0_layer_call_fn_85494bqnpo7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
A__inference_gnn_63_layer_call_and_return_conditional_losses_83699�<=GHEFQR\]Z[z{pqno����x�u
n�k
a�^
?�<
environment+���������������������������
�
state���������
p

 
� "b�_
X�U
$�!

tensor_0_0���������
-�*

tensor_0_1������������������
� �
A__inference_gnn_63_layer_call_and_return_conditional_losses_83914�<=HEGFQR]Z\[z{qnpo����x�u
n�k
a�^
?�<
environment+���������������������������
�
state���������
p 

 
� "b�_
X�U
$�!

tensor_0_0���������
-�*

tensor_0_1������������������
� �
A__inference_gnn_63_layer_call_and_return_conditional_losses_84847�<=GHEFQR\]Z[z{pqno����x�u
n�k
a�^
<�9
inputs_0+���������������������������
�
inputs_1���������
p

 
� "b�_
X�U
$�!

tensor_0_0���������
-�*

tensor_0_1������������������
� �
A__inference_gnn_63_layer_call_and_return_conditional_losses_85070�<=HEGFQR]Z\[z{qnpo����x�u
n�k
a�^
<�9
inputs_0+���������������������������
�
inputs_1���������
p 

 
� "b�_
X�U
$�!

tensor_0_0���������
-�*

tensor_0_1������������������
� �
&__inference_gnn_63_layer_call_fn_84062�<=GHEFQR\]Z[z{pqno����x�u
n�k
a�^
?�<
environment+���������������������������
�
state���������
p

 
� "T�Q
"�
tensor_0���������
+�(
tensor_1�������������������
&__inference_gnn_63_layer_call_fn_84209�<=HEGFQR]Z\[z{qnpo����x�u
n�k
a�^
?�<
environment+���������������������������
�
state���������
p 

 
� "T�Q
"�
tensor_0���������
+�(
tensor_1�������������������
&__inference_gnn_63_layer_call_fn_84530�<=GHEFQR\]Z[z{pqno����x�u
n�k
a�^
<�9
inputs_0+���������������������������
�
inputs_1���������
p

 
� "T�Q
"�
tensor_0���������
+�(
tensor_1�������������������
&__inference_gnn_63_layer_call_fn_84582�<=HEGFQR]Z\[z{qnpo����x�u
n�k
a�^
<�9
inputs_0+���������������������������
�
inputs_1���������
p 

 
� "T�Q
"�
tensor_0���������
+�(
tensor_1�������������������
F__inference_legals_mask_layer_call_and_return_conditional_losses_85468�}�z
s�p
n�k
8�5
inputs_0'���������������������������
/�,
inputs_1������������������
� "9�6
/�,
tensor_0������������������
� �
+__inference_legals_mask_layer_call_fn_85457�}�z
s�p
n�k
8�5
inputs_0'���������������������������
/�,
inputs_1������������������
� ".�+
unknown�������������������
H__inference_policy_logits_layer_call_and_return_conditional_losses_85610�z{���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp"9�6
/�,
tensor_0������������������
� �
H__inference_policy_logits_layer_call_and_return_conditional_losses_85650�z{���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp "9�6
/�,
tensor_0������������������
� �
-__inference_policy_logits_layer_call_fn_85559�z{���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp".�+
unknown�������������������
-__inference_policy_logits_layer_call_fn_85570�z{���
���
���
8�5
inputs_0'���������������������������
8�5
inputs_1'���������������������������
/�,
inputs_2������������������
�

trainingp ".�+
unknown�������������������
I__inference_policy_targets_layer_call_and_return_conditional_losses_85706�p�m
f�c
]�Z
+�(
inputs_0������������������
+�(
inputs_1������������������


 
� "5�2
+�(
tensor_0������������������
� �
.__inference_policy_targets_layer_call_fn_85696�p�m
f�c
]�Z
+�(
inputs_0������������������
+�(
inputs_1������������������


 
� "*�'
unknown�������������������
E__inference_reshape_65_layer_call_and_return_conditional_losses_85087[+�(
!�
�
inputs���������
� ",�)
"�
tensor_0���������
� ~
*__inference_reshape_65_layer_call_fn_85075P+�(
!�
�
inputs���������
� "!�
unknown����������
#__inference_signature_wrapper_84478�<=HEGFQR]Z\[z{qnpo�������
� 
y�v
N
environment?�<
environment+���������������������������
$
state�
state���������"��
C
policy_targets1�.
policy_targets������������������
8
value_targets'�$
value_targets����������
H__inference_value_targets_layer_call_and_return_conditional_losses_85690f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
-__inference_value_targets_layer_call_fn_85679[��0�-
&�#
!�
inputs����������
� "!�
unknown���������