import { BrowserRouter as Routers, Routes, Route } from 'react-router-dom'
import GoldHomePage from './Pages/GoldHomePage'

function App() {

  return (
    <Routers>
      <Routes>
        <Route path='/' element={<GoldHomePage/>}/>
      </Routes>
    </Routers>
  )
}

export default App
