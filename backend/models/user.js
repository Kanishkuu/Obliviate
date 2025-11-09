import mongoose from 'mongoose';
const UserSchema = new mongoose.Schema({
  googleSub: { type: String, unique: true, index: true },
  name: String,
  email: { type: String, index: true },
  avatar: String,
}, { timestamps: true });
export default mongoose.model('User', UserSchema);
