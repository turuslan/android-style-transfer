package tf.style_transfer

import android.app.Application
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.DequantizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.io.InputStream

fun InputStream.bmp(): ImageBitmap {
    val bmp = BitmapFactory.decodeStream(this)
    this.close()
    return bmp.asImageBitmap()
}

fun ImageBitmap.jpg(): ByteArray {
    val s = ByteArrayOutputStream()
    asAndroidBitmap().compress(Bitmap.CompressFormat.JPEG, 80, s)
    return s.toByteArray()
}

fun Context.assetBmp(path: String): ImageBitmap {
    return assets.open(path).bmp()
}

fun Context.assetModel(path: String): Interpreter {
    return Interpreter(FileUtil.loadMappedFile(this, path), Interpreter.Options())
}

fun ImageBitmap.tensor(shape: IntArray): TensorImage {
    val imageProcessor =
        ImageProcessor.Builder().add(ResizeOp(shape[1], shape[2], ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)).build()
    val tensorImage = TensorImage(DataType.FLOAT32)
    tensorImage.load(this.asAndroidBitmap())
    return imageProcessor.process(tensorImage)
}

fun TensorBuffer.bmp(aspect: Float): ImageBitmap {
    var w = shape[2]
    var h = w
    if (aspect < 1) {
        w = (h.toFloat() * aspect).toInt()
    } else {
        h = (w.toFloat() / aspect).toInt()
    }
    val imagePostProcessor = ImageProcessor.Builder().add(DequantizeOp(0f, 255f))
        .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR)).build()
    val tensorImage = TensorImage(DataType.FLOAT32)
    tensorImage.load(this)
    return imagePostProcessor.process(tensorImage).bitmap.asImageBitmap()
}

class Samples(context: Context) {
    val image1 = context.assetBmp("image1.jpg")
    val style1 = context.assetBmp("style1.jpg")
}

class Model(context: Context) {
    private val predict = context.assetModel("predict_int8.tflite")
    private val transfer = context.assetModel("transfer_int8.tflite")

    private val predictShapeIn: IntArray = predict.getInputTensor(0).shape()
    private val predictShapeOut: IntArray = predict.getOutputTensor(0).shape()
    private val transferShapeIn: IntArray = transfer.getInputTensor(0).shape()
    private val transferShapeOut: IntArray = transfer.getOutputTensor(0).shape()

    private var styleTensorCache: TensorBuffer? = null
    private val output = TensorBuffer.createFixedSize(transferShapeOut, DataType.FLOAT32)

    fun merge(content: ImageBitmap, style: ImageBitmap): ImageBitmap {
        val styleTensor: TensorBuffer =
            styleTensorCache ?: TensorBuffer.createFixedSize(predictShapeOut, DataType.FLOAT32)
                .also {
                    predict.run(style.tensor(predictShapeIn).buffer, it.buffer)
                    styleTensorCache = it
                }
        transfer.runForMultipleInputsOutputs(
            arrayOf(content.tensor(transferShapeIn).buffer, styleTensor.buffer),
            mapOf(Pair(0, output.buffer)),
        )
        return output.bmp(content.width.toFloat() / content.height.toFloat())
    }

    fun resetStyle() {
        styleTensorCache = null
    }
}

class MyViewModel(app: Application) : AndroidViewModel(app) {
    private val samples = Samples(app)
    val left = mutableStateOf(samples.image1)
    val right = mutableStateOf(samples.style1)
    val result = mutableStateOf<ImageBitmap?>(null)
    private val model = Model(app)
    private var job: Job? = null
    private var dirty = false

    fun merge() {
        result.value = null
        dirty = true
        if (job != null) {
            return
        }
        job = viewModelScope.launch(Dispatchers.IO) {
            var img: ImageBitmap? = null
            while (dirty) {
                dirty = false
                img = model.merge(left.value, right.value)
            }
            result.value = img
            job = null
        }
    }

    fun styleChanged() {
        model.resetStyle()
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val vm = viewModel<MyViewModel>()
            LaunchedEffect(vm.right.value) { vm.styleChanged() }
            LaunchedEffect(vm.left.value, vm.right.value) { vm.merge() }
            Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                App(vm.left, vm.right, vm.result, modifier = Modifier.padding(innerPadding))
            }
        }
    }
}

val TILE = 120.dp

@Composable
fun App(
    left: MutableState<ImageBitmap>,
    right: MutableState<ImageBitmap>,
    result: State<ImageBitmap?>,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    Column(modifier = modifier.fillMaxSize()) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        ) {
            result.value?.let {
                Image(it, "", modifier = Modifier.fillMaxSize())
                FloatingActionButton(
                    onClick = {
                        val name = "share.jpg"
                        context.openFileOutput(name, 0).apply {
                            write(it.jpg())
                            flush()
                            close()
                        }
                        val file = context.getFileStreamPath(name)
                        val uri = FileProvider.getUriForFile(context, "tf.style_transfer", file)
                        context.startActivity(Intent.createChooser(Intent().apply {
                            action = Intent.ACTION_SEND
                            putExtra(Intent.EXTRA_STREAM, uri)
                            type = "image/jpg"
                        }, null))
                    }, modifier = Modifier
                        .padding(8.dp)
                        .align(Alignment.TopEnd)
                ) {
                    Icon(Icons.Filled.Share, "")
                }
            }
        }
        Row(
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Tile(left)
            Tile(right)
        }
    }
}

@Composable
fun Tile(img: MutableState<ImageBitmap>) {
    val context = LocalContext.current
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) {
        it?.let {
            context.contentResolver.openInputStream(it)?.bmp()?.let { img.value = it }
        }
    }
    Image(img.value, "", modifier = Modifier
        .size(TILE)
        .clickable {
            picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        })
}

@Preview(showBackground = true, widthDp = 340, heightDp = 600)
@Composable
fun AppPreview() {
    val context = LocalContext.current
    val samples = remember { Samples(context) }
    val left = remember { mutableStateOf(samples.image1) }
    val right = remember { mutableStateOf(samples.style1) }
    App(left, right, left)
}
