const http = require('http');
const express = require('express')
const app = express()


app.use(express.static(__dirname + '/public'))

const hostname = '127.0.0.1';
const port = 3000;

app.listen(port, hostname, () => {
  console.log(`El servidor se está ejecutando en http://${hostname}:${port}/`);
});