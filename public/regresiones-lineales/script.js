let stopTraining

const getData = async () => {
  const datosCasaR = await fetch("datos.json")
  const datosCasas = await datosCasaR.json()
  let datosLimpios = datosCasas.map((casa) => ({
    precio: casa.Precio,
    cuartos: casa.NumeroDeCuartosPromedio,
  }))
  datosLimpios = datosLimpios.filter(
    (casa) => casa.precio != null && casa.cuartos != null
  )

  return datosLimpios
}

const visualizarDatos = (data) => {
  const valores = data.map((d) => ({ x: d.cuartos, y: d.precio }))

  tfvis.render.scatterplot(
    { name: "Cuartos vs Precios" },
    { values: valores },
    { xLabel: "Cuartos", yLabel: "Precios", height: 300 }
  )
}

const crearModelo = () => {
  const modelo = tf.sequential()
  modelo.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))
  modelo.add(tf.layers.dense({ units: 1, units: 1, useBias: true }))
  return modelo
}

const optimizador = tf.train.adam()
const funcion_perdida = tf.losses.meanSquaredError
const metricas = ["mse"]

const entrenarmodelo = async (model, inputs, labels) => {
  // Prepare the model for training.
  model.compile({
    optimizer: optimizador,
    loss: funcion_perdida,
    metrics: metricas,
  })

  const surface = { name: "show.history live", tab: "Training" }
  const tamanioBatch = 28
  const epochs = 50
  const history = []

  return await model.fit(inputs, labels, {
    tamanioBatch,
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epch, log) => {
        history.push(log)
        tfvis.show.history(surface, history, ["loss", "mse"])
        if (stopTraining) {
          modelo.stopTraining = true
        }
      },
    },
  })
}

//mostrar curva de inferencia()
const verCurvaInferencia = async () => {
  var data = await getData()
  var tensorData = await convertirDatosATensores(data)

  const { entradasMax, entradasMin, etiquetasMin, etiquetasMax } = tensorData

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100)
    const preds = modelo.predict(xs.reshape([100, 1]))

    const desnormX = xs.mul(entradasMax.sub(entradasMin)).add(entradasMin)

    const desnormY = preds.mul(etiquetasMax.sub(etiquetasMin)).add(etiquetasMin)

    return [desnormX.dataSync(), desnormY.dataSync()]
  })

  const puntosPrediccion = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] }
  })

  const puntosOriginales = data.map((d) => ({
    x: d.cuartos,
    y: d.precio,
  }))

  tfvis.render.scatterplot(
    { name: "Prediccion vs Originales" },
    {
      values: [puntosOriginales, puntosPrediccion],
      series: ["originales", "predicciones"],
    },
    {
      xLabel: "Cuartos",
      yLabel: "Precio",
      height: 300,
      width: 600,
    }
  )
}

const cargarModelo = async () => {
  const uploadJSONInput = document.getElementById("upload-json")
  const uploadWeightsInput = document.getElementById("upload-weights")

  modelo = await tf.loadLayersModel(
    tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]])
  )
  console.log("Modelo Cargado")
}

async function guardarModelo() {
  const saveResult = await modelo.save("downloads://modelo-regresion")
}

const convertirDatosATensores = (data) => {
  return tf.tidy(() => {
    tf.util.shuffle(data)

    const entradas = data.map((d) => d.cuartos)
    const etiquetas = data.map((d) => d.precio)

    const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1])
    const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1])

    const entradasMax = tensorEntradas.max()
    const entradasMin = tensorEntradas.min()
    const etiquetasMax = tensorEtiquetas.max()
    const etiquetasMin = tensorEtiquetas.min()

    // (dato -min) / (max-min)
    const entradasNormalizadas = tensorEntradas
      .sub(entradasMin)
      .div(entradasMax.sub(entradasMin))
    const etiquetasNormalizadas = tensorEtiquetas
      .sub(etiquetasMin)
      .div(etiquetasMax.sub(etiquetasMin))

    return {
      entradas: entradasNormalizadas,
      etiquetas: etiquetasNormalizadas,
      // Return the min/max bounds so we can use them later.
      entradasMax,
      entradasMin,
      etiquetasMax,
      etiquetasMin,
    }
  })
}

let modelo

const run = async () => {
  const data = await getData()
  visualizarDatos(data)

  modelo = crearModelo()
  const tensorData = convertirDatosATensores(data)
  const { entradas, etiquetas } = tensorData

  await entrenarmodelo(modelo, entradas, etiquetas)
}

run()
