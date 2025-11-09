import AuthProvider from "./auth/authcontext"
import ProtectedRoute from "./auth/ProtectedRoute.jsx"
import { Route, Routes } from "react-router-dom";
import Login from "./pages/Login.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Assets from "./pages/Assets.jsx";
import Unlearn from "./pages/Unlearn.jsx";
import Jobs from "./pages/Jobs.jsx";
import Auditor from "./pages/Auditor.jsx";
import Unlearning from "./pages/Unlearning.jsx";
import Finetune from "./pages/finetune.jsx";


function App() {

  return (
    <>
      <div>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login/>} />
          <Route path="/" element={<ProtectedRoute><Dashboard/></ProtectedRoute>} />
          <Route path="/assets" element={<ProtectedRoute><Assets/></ProtectedRoute>} />
          <Route path="/auditor" element={<ProtectedRoute><Auditor/></ProtectedRoute>} />
          <Route path="/unlearn" element={<ProtectedRoute><Unlearn/></ProtectedRoute>} />
          <Route path="/unlearning" element={<ProtectedRoute><Unlearning/></ProtectedRoute>} />
          <Route path="/finetune" element={<ProtectedRoute><Finetune/></ProtectedRoute>} />
          <Route path="/jobs" element={<ProtectedRoute><Jobs/></ProtectedRoute>} />
        </Routes>
      </AuthProvider>
    </div>
    </>
  )
}

export default App
