import jwt from "jsonwebtoken";

export default function auth(req, res, next) {
  const token = req.cookies?.obl_sess;
  if (!token) return res.status(401).json({ error: "Unauthorized" });

  try {
    const { uid } = jwt.verify(token, process.env.JWT_SECRET);
    req.userId = uid;
    next();
  } catch {
    return res.status(401).json({ error: "Unauthorized" });
  }
}
