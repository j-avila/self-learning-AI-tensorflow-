let net

let cam = true

const imgEl = document.getElementById("img")
const descEl = document.getElementById("descripcion")
const webCamElement = document.getElementById("webcam")
const classifier = knnClassifier.create()

const app = async () => {
  net = await mobilenet.load()

  let result = await net.classify(imgEl)
  console.log(result)
  displayImagePrediction()

  webcam = await tf.data.webcam(webCamElement)
}
const startScan = async (active) => {
  cam = active
  while (cam) {
    const img = await webcam.capture()
    const result = await net.classify(img)
    const activation = net.infer(img, "conv_preds") //compartiendo conocimiento para entrenar al modelo

    let result2
    const classes = [
      "untrained",
      "teclado",
      "pollo",
      "vaso",
      "jose",
      "fuck you!",
      "ok",
    ]

    try {
      result2 = await classifier.predictClass(activation)
      document.getElementById("console2").innerText = `
            prediction: ${classes[result2.label]}\n
            probability: ${result2.confidences[result2.label]}
          `
    } catch {
      console.log("modelo no configurado")
    }

    document.getElementById("console").innerHTML = `
        prediction: ${result[0].className} 
        probability: ${result[0].probability}
        `
    img.dispose()
    await tf.nextFrame()
  }
}

imgEl.onload = async () => {
  displayImagePrediction()
}

const addExample = async (classId) => {
  const img = await webcam.capture()
  const activation = net.infer(img, true)

  classifier.addExample(activation, classId)

  img.dispose()
}

const displayImagePrediction = async () => {
  try {
    let result = await net.classify(imgEl)
    descEl.innerHTML = JSON.stringify(result)
  } catch (err) {
    console.log(err)
  }
}

count = 0

const cambiarimagen = () => {
  count = count + 1
  imgEl.src = "https://picsum.photos/200/300?random=" + count
}

app()
