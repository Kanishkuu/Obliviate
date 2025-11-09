import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import cookieParser from 'cookie-parser';
import mongoose from 'mongoose';
import helmet from "helmet";
import uploadsRouter from "./routes/uploads.js";
import jobsRouter from "./routes/job.js";
import finetuneRoutes from "./routes/finetune.js";
import modelsRoute from "./routes/model.js";
import auditorRoute from "./routes/auditor.js";
import unlearnRoute from "./routes/unlearn.js";

const app = express();
app.use(cors({
  origin: "http://localhost:5173",
  credentials: true,
  methods: ["GET","POST","PUT","DELETE","OPTIONS"],
  allowedHeaders: ["Content-Type","Authorization"]
}));




app.use(
  helmet({
    crossOriginOpenerPolicy: { policy: "same-origin-allow-popups" },
    crossOriginEmbedderPolicy: false,  
  })
);

app.use(express.json());
app.use(cookieParser());

// DB
mongoose.connect(process.env.MONGO_URI).then(()=>console.log('Mongo connected')).catch(console.error);

app.use("/api/finetune", finetuneRoutes);
app.use("/api", uploadsRouter);
app.use("/api", jobsRouter);

// health
app.get('/health', (req,res)=>res.json({ ok:true }));


app.use("/api/models", modelsRoute);
app.use("/api/auditor", auditorRoute);
app.use("/api/unlearn", unlearnRoute);

// routes
import authRouter from './routes/auth.js';
app.use('/auth', authRouter);


const port = process.env.PORT || 5000;
app.listen(port, ()=>console.log(`API on :${port}`));
