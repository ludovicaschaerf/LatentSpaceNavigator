import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import React, { useRef, useState, useEffect } from "react";
import { Vector3 } from "three";
import * as THREE from 'three';
import { OrbitControls, Bounds, Box } from "@react-three/drei";
import { useStore } from "./store/useStore.jsx";

import Info from "./Info.jsx";
import Pivot from "./Pivot.jsx";
import Cloud from "./Cloud.jsx";
import Arrow from "./Arrow.jsx";


export default function App() {
    const position = useStore((state) => state.position);
    
    const vectors = useStore((state) => state.vectors);
    const [imageData, setImageData] = useState('');
    const [oldPos, setoldPos] = useState('');
    const oldposition = useStore((state) => state.oldposition);
    const colorclicked = useStore((state) => state.colorclicked);
    
    useEffect(() => {
        async function fetchImage() {
            try {
                const response = await fetch('http://127.0.0.1:5000/get-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify([colorclicked, oldposition]),
                });
                const data = await response.json();
                setImageData(data.imageData);
                setoldPos(data.newPosition);
            } catch (error) {
                console.error('Error fetching image:', error);
            }
        }

        fetchImage();
    }, [colorclicked, oldposition]);


    return (
        <>
            <Info />
            <Canvas>
                <OrbitControls />
                <ambientLight intensity={1} />
                <pointLight position={[20, 20, 20]} />

                <Bounds
                    fit
                    margin={3}
                >
                    <Pivot position={position} imageData={imageData} />
                    {Object.keys(vectors).map((vec, index) => (
                        <Arrow
                            position={position}
                            index={index}
                            direction={vectors[vec].direction}
                            color={vectors[vec].color}
                            colorname={vec}
                            oldpos={oldPos}
                            scale={0.5}
                        />
                    ))}
                </Bounds>
                
                <Cloud />
            </Canvas>
            
        </>
    );

}
