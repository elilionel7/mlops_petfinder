const express = require('express');
const app = express();
const port = 3001;
const routes = require('./routes');
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Initialize routes
routes(app);

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
