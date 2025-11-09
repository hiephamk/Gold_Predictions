import { useState, useEffect } from 'react';
import FxPredictionResult from '../components/GoldPrediction/FxPredictionResult'
import { Box, Button, Center, Container, Heading, HStack, VStack } from '@chakra-ui/react'
import FxActualPredictedData from '../components/GoldPrediction/FxActualPredictedData';
import GoldActualData from '../components/GoldPrediction/GoldActualData';

const getInitialInterval = () => {
  return (
    (localStorage.getItem("fx-default-interval") as
      | '15m'
      | '30m'
      | '1h'
      | '4h'
      | '1d'
      | '1wk') ?? '1h'
  );
};
const GoldHomePage: React.FC = () => {
  const [interval, setPredictionInterval] = useState<'15m'| '30m'| '1h'| '4h'|'1d' | '1wk'>(getInitialInterval);
  
  const saveDefault = () => {
    localStorage.setItem("fx-default-interval", interval)
    alert("✅ Defaults saved!");
  }
  
  return (
    <Container w={"1100px"}>
      <Center>
        <VStack>
          <Heading my={"20px"} fontSize="36px" fontWeight="bold">Gold Price Prediction</Heading>
          <HStack shadow="3px 3px 15px 5px rgb(75, 75, 79)" p={"10px"} rounded={"5px"}>
            <form>
              <label>Prediction Interval</label>
              <select
                value={interval}
                onChange={(e) => setPredictionInterval(e.target.value as '15m'| '30m'| '1h'| '4h'| '1d' | '1wk')}
                style={{border:'1px solid', borderRadius:'3px', margin:'10px', padding:'5px'}}
              >
                <option value="15m">15 Minutes (15m)</option>
                <option value="30m">30 Minutes (30m)</option>
                <option value="1h">1 Hour (1h)</option>
                <option value="4h">4 Hours (4h)</option>
                <option value="1d">Daily (1d)</option>
                <option value="1wk">Weekly (1wk)</option>
              </select>
            </form>
            <Button onClick={saveDefault}>set default</Button>
          </HStack>
        </VStack>
      </Center>
        
      <HStack h={"fit-content"} >
          <Box w={"100%"}>
            <FxPredictionResult interval={interval}/>
            {/* <Box rounded={"7px"} shadow="3px 3px 15px 5px rgb(75, 75, 79)"><FxPredictionResult_Chart interval={interval}/></Box> */}
            <Box rounded={"7px"} shadow="3px 3px 15px 5px rgb(75, 75, 79)"><GoldActualData interval={interval}/></Box>
            <Box my={"10px"} rounded={"7px"} shadow="3px 3px 15px 5px rgb(75, 75, 79)"><FxActualPredictedData interval={interval}/></Box>
          </Box>
      </HStack>
    </Container>
  )
}

export default GoldHomePage