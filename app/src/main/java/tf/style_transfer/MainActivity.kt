package tf.style_transfer

import android.content.res.AssetManager
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import tf.style_transfer.ui.theme.StyleTransferTheme

fun assetBmp(assets: AssetManager, path: String): ImageBitmap {
    val s = assets.open(path)
    val bmp = BitmapFactory.decodeStream(s)
    s.close()
    return bmp.asImageBitmap()
}

class Samples(assets: AssetManager) {
    val image1 = assetBmp(assets, "image1.jpg")
    val style1 = assetBmp(assets, "style1.jpg")
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val assets = LocalContext.current.assets
            val samples = remember { Samples(assets) }
            StyleTransferTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    App(
                        samples, modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

val TILE = 120.dp

@Composable
fun App(samples: Samples, modifier: Modifier = Modifier) {
    Column(modifier = modifier.fillMaxSize()) {
        Image(
            samples.image1, "", modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        )
        Row(
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Tile(samples.image1)
            Tile(samples.style1)
        }
    }
}

@Composable
fun Tile(bmp: ImageBitmap) {
    Image(bmp, "", modifier = Modifier.size(TILE))
}

@Preview(showBackground = true, widthDp = 340, heightDp = 600)
@Composable
fun AppPreview() {
    val samples = Samples(LocalContext.current.assets)
    StyleTransferTheme {
        App(samples)
    }
}
