import { Router } from 'express';
import jwt from 'jsonwebtoken';
import User from '../models/user.js';
import { OAuth2Client } from 'google-auth-library';
import auth from '../middleware/auth.js';

const router = Router();
const client = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);

router.post('/google', async (req,res)=>{
  try {
    const { credential } = req.body;

    if (!credential) {

      return res.status(400).json({ error: 'Missing credential' });
    }

    const ticket = await client.verifyIdToken({
      idToken: credential,
      audience: process.env.GOOGLE_CLIENT_ID
    });

    const p = ticket.getPayload();
  

    let user = await User.findOne({ googleSub: p.sub });
  

    if (!user) {

      user = await User.create({ googleSub: p.sub, email: p.email, name: p.name, avatar: p.picture });
    }



    const token = jwt.sign({ uid: user._id }, process.env.JWT_SECRET, { expiresIn: '7d' });
 

    res.cookie('obl_sess', token, {
      httpOnly: true, sameSite: 'lax', secure: false, maxAge: 7*24*60*60*1000
    });

 
    return res.json({ ok: true, user });

  } catch (e) {

    return res.status(401).json({ error: 'Invalid Google token' });
  }
});


// whoami from cookie
router.get("/me", auth, async (req, res) => {
  const u = await User.findById(req.userId).select("_id name email avatar");
  if (!u) return res.status(404).json({ error: "Not found" });
  res.json({
    ok: true,
    user: { id: u._id, name: u.name, email: u.email, avatar: u.avatar }
  });
});



router.post('/logout', (req,res)=>{
  res.clearCookie('obl_sess');
  res.json({ ok:true });
});

export default router;
